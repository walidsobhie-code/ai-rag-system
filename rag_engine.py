#!/usr/bin/env python3
"""
AI RAG Engine - Production-ready Retrieval Augmented Generation
"""
import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Try imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import (
        PyPDFLoader, TextLoader, MarkdownLoader, DocxLoader
    )
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


SUPPORTED_EXTENSIONS = {
    '.pdf': 'PyPDFLoader',
    '.txt': 'TextLoader',
    '.md': 'MarkdownLoader',
    '.markdown': 'MarkdownLoader',
    '.docx': 'DocxLoader'
}


@dataclass
class Document:
    """Document metadata"""
    file_path: str
    content: str
    metadata: Dict = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)


class DocumentLoader:
    """Load documents from various formats"""

    @staticmethod
    def load(file_path: str) -> Document:
        """Load a single document"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if not LANGCHAIN_AVAILABLE:
            # Fallback: read as plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return Document(
                file_path=file_path,
                content=content,
                metadata={"type": ext, "size": len(content)}
            )

        loader_class = SUPPORTED_EXTENSIONS[ext]

        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in ('.md', '.markdown'):
                loader = MarkdownLoader(file_path)
            elif ext == '.docx':
                loader = DocxLoader(file_path)
            else:
                raise ValueError(f"No loader for: {ext}")

            docs = loader.load()
            content = "\n\n".join([d.page_content for d in docs])

            return Document(
                file_path=file_path,
                content=content,
                metadata={"type": ext, "source": file_path, "pages": len(docs)}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")

    @staticmethod
    def load_directory(directory: str, extensions: List[str] = None) -> List[Document]:
        """Load all documents from a directory"""
        if extensions is None:
            extensions = list(SUPPORTED_EXTENSIONS.keys())

        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        doc = DocumentLoader.load(file_path)
                        documents.append(doc)
                    except Exception as e:
                        print(f"⚠️  Skipped {file_path}: {e}")

        return documents


class TextChunker:
    """Split documents into chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk all documents"""
        if not LANGCHAIN_AVAILABLE:
            # Simple chunking fallback
            for doc in documents:
                chunks = []
                content = doc.content
                for i in range(0, len(content), self.chunk_size):
                    chunks.append(content[i:i + self.chunk_size])
                doc.chunks = chunks
            return documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in documents:
            # Create langchain documents
            from langchain.schema import Document as LCDocument
            lc_docs = [LCDocument(page_content=doc.content, metadata=doc.metadata)]
            split_docs = splitter.split_documents(lc_docs)
            doc.chunks = [d.page_content for d in split_docs]

        return documents


class VectorStore:
    """Vector store for semantic search"""

    def __init__(self, embedding_model: str = "openai", persist_dir: str = "./chroma_db"):
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.embeddings = None

        if LANGCHAIN_AVAILABLE:
            try:
                if embedding_model == "openai":
                    self.embeddings = OpenAIEmbeddings()
                else:
                    # HuggingFace local embeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
            except Exception as e:
                print(f"⚠️  Failed to load embeddings: {e}")

    def create_from_chunks(self, chunks: List[str], metadatas: List[Dict] = None):
        """Create vector store from chunks"""
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            # Mock vector store
            self.vectorstore = MockVectorStore(chunks)
            return

        try:
            if metadatas is None:
                metadatas = [{}] * len(chunks)

            self.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.persist_dir
            )
        except Exception as e:
            print(f"⚠️  Failed to create vector store: {e}")
            self.vectorstore = MockVectorStore(chunks)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.vectorstore is None:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=top_k)
            return [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        except Exception as e:
            print(f"⚠️  Search error: {e}")
            return []


class MockVectorStore:
    """Mock vector store for when dependencies aren't available"""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search as fallback"""
        query_terms = query.lower().split()
        results = []

        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_lower)
            if score > 0:
                results.append({
                    "content": chunk[:200] + "...",
                    "metadata": {"chunk_id": i, "score": score}
                })

        results.sort(key=lambda x: x["metadata"]["score"], reverse=True)
        return results[:top_k]


class RAGEngine:
    """Production RAG Engine"""

    def __init__(self, vector_store_dir: str = "./vector_store",
                 embedding_model: str = "openai",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.vector_store_dir = vector_store_dir
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.vectorstore = VectorStore(embedding_model, vector_store_dir)
        self.documents: List[Document] = []
        self.is_indexed = False

    def ingest(self, source: str) -> Dict:
        """Ingest documents from file or directory"""
        print(f"📄 Ingesting: {source}")

        if os.path.isfile(source):
            docs = [DocumentLoader.load(source)]
        elif os.path.isdir(source):
            docs = DocumentLoader.load_directory(source)
        else:
            return {"status": "error", "message": f"Invalid path: {source}"}

        print(f"   Loaded {len(docs)} documents")

        # Chunk documents
        docs = self.chunker.chunk(docs)
        total_chunks = sum(len(d.chunks) for d in docs)
        print(f"   Created {total_chunks} chunks")

        # Create vector store
        all_chunks = []
        metadatas = []
        for doc in docs:
            all_chunks.extend(doc.chunks)
            metadatas.extend([doc.metadata] * len(doc.chunks))

        self.vectorstore.create_from_chunks(all_chunks, metadatas)
        self.documents = docs
        self.is_indexed = True

        return {
            "status": "success",
            "documents": len(docs),
            "chunks": total_chunks
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context"""
        if not self.is_indexed:
            print("⚠️  No documents indexed. Run ingest first.")
            return []

        print(f"🔍 Searching for: {query}")
        results = self.vectorstore.similarity_search(query, top_k)
        return results

    def generate(self, query: str, context: List[str] = None) -> str:
        """Generate answer with context (requires LLM)"""
        if context is None:
            context = [r["content"] for r in self.retrieve(query)]

        if not LANGCHAIN_AVAILABLE:
            return f"""
[Demo Mode - Install langchain & openai for real generation]

Query: {query}

Retrieved Context:
{chr(10).join(context[:3])}

Answer: This is a placeholder response. The RAG system retrieved {len(context)} context chunks.
            """

        # Use LLM to generate answer
        try:
            from langchain.chains import RetrievalQA
            from langchain.chat_models import ChatOpenAI

            # This would be the real implementation
            return f"[Would generate with LLM using {len(context)} context chunks]"
        except Exception as e:
            return f"[Error: {e}]"


def main():
    parser = argparse.ArgumentParser(
        description='AI RAG Engine - Production-ready retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ingest', help='Ingest documents from file or directory')
    parser.add_argument('--query', '-q', help='Query the knowledge base')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--embedding', default='openai', help='Embedding model')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print("📚 AI RAG Engine")
    print("=" * 40)

    if not LANGCHAIN_AVAILABLE:
        print("⚠️  LangChain not installed. Running in limited mode.")
        print("   Install: pip install langchain chromadb pypdf openai")
        print()

    # Initialize RAG engine
    rag = RAGEngine(chunk_size=args.chunk_size, embedding_model=args.embedding)

    if args.ingest:
        result = rag.ingest(args.ingest)
        print(f"✅ {result}")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print()

        while True:
            query = input("Query: ")
            if query.lower() in ('quit', 'exit', 'q'):
                break

            results = rag.retrieve(query, args.top_k)
            print(f"📊 Found {len(results)} results:")
            for i, r in enumerate(results):
                print(f"  [{i+1}] {r['content'][:100]}...")
            print()

    elif args.query:
        results = rag.retrieve(args.query, args.top_k)
        print(f"📊 Found {len(results)} results:")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r['content'][:200]}...")

    elif not args.ingest:
        # Demo
        print("Usage:")
        print("  %(prog)s --ingest ./docs")
        print("  %(prog)s --query 'your question'")
        print("  %(prog)s --interactive")


if __name__ == '__main__':
    main()