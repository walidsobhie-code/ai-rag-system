#!/usr/bin/env python3
"""
AI RAG System - Production RAG with LangChain
Retrieval Augmented Generation for enterprise knowledge bases
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()

class RAGEngine:
    """Production RAG Engine with LangChain"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        
        if LANGCHAIN_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                self.llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=api_key)
                self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load existing vector store or create new one"""
        if os.path.exists(self.persist_directory) and self.embeddings:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded existing vector store with {self.vectorstore._collection.count()} documents")
    
    def ingest(self, file_path: str) -> Dict:
        """Ingest a document into the knowledge base"""
        if not LANGCHAIN_AVAILABLE:
            return {"status": "needs_langchain", "chunks": 0}
        
        if not self.embeddings:
            return {"status": "needs_api_key", "chunks": 0}
        
        path = Path(file_path)
        if not path.exists():
            return {"status": "file_not_found", "chunks": 0}
        
        print(f"📄 Ingesting: {file_path}")
        
        # Load document
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            loader = TextLoader(str(path))
        
        documents = loader.load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        self.vectorstore.persist()
        
        print(f"✅ Ingested {len(chunks)} chunks from {path.name}")
        return {"status": "success", "chunks": len(chunks), "file": path.name}
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context for a query"""
        if not self.vectorstore:
            return [{"text": "Vector store not initialized", "score": 0}]
        
        docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs:
            results.append({
                "text": doc.page_content[:500],
                "score": float(score),
                "source": doc.metadata.get("source", "unknown")
            })
        
        return results
    
    def generate(self, query: str, top_k: int = 5) -> str:
        """Generate answer with RAG"""
        if not self.vectorstore or not self.llm:
            results = self.retrieve(query, top_k)
            if results:
                context = "\n\n".join([r["text"] for r in results])
                return f"Based on the knowledge base:\n\n{context[:1000]}...\n\n(RAG requires OPENAI_API_KEY)"
            return "No context found. Please ingest documents first."
        
        # Create retrieval QA chain
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        return result["result"]
    
    def chat(self, query: str) -> Dict:
        """Chat with the knowledge base"""
        results = self.retrieve(query)
        answer = self.generate(query)
        return {
            "answer": answer,
            "sources": results
        }

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI RAG System")
    parser.add_argument("--ingest", help="Document to ingest")
    parser.add_argument("--query", help="Query to answer")
    parser.add_argument("--chat", help="Chat with knowledge base")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()
    
    rag = RAGEngine()
    
    if args.ingest:
        result = rag.ingest(args.ingest)
        print(json.dumps(result, indent=2))
    
    elif args.query:
        result = rag.chat(args.query)
        print(f"\n💬 Answer:\n{result['answer']}\n")
        print(f"📚 Sources: {len(result['sources'])} documents found")
    
    elif args.chat:
        print("💬 Chat mode - Type 'exit' to quit")
        while True:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                break
            result = rag.chat(query)
            print(f"\nAI: {result['answer'][:500]}")
    
    else:
        print("""
🤖 AI RAG System

Usage:
  --ingest <file>   Ingest a document (PDF, DOCX, TXT)
  --query <text>    Query the knowledge base
  --chat           Interactive chat mode

Example:
  python rag_engine.py --ingest ./docs/report.pdf
  python rag_engine.py --query "What is the main topic?"
        """)

if __name__ == "__main__":
    main()
