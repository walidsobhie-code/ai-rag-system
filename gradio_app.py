#!/usr/bin/env python3
"""
AI RAG System - Gradio Chat Interface
Production-ready web UI for the RAG engine
"""
import os
import sys
import gradio as gr
from rag_engine import RAGEngine

# Global RAG engine instance
rag_engine = None
is_indexed = False


def initialize_rag(ingest_path, chunk_size=1000, embedding_model="openai"):
    """Initialize the RAG engine"""
    global rag_engine, is_indexed

    if not ingest_path:
        return "⚠️  Please provide a path to ingest"

    if not os.path.exists(ingest_path):
        return f"⚠️  Path does not exist: {ingest_path}"

    try:
        rag_engine = RAGEngine(
            chunk_size=chunk_size,
            embedding_model=embedding_model
        )
        result = rag_engine.ingest(ingest_path)
        is_indexed = True
        return f"✅ Indexed {result.get('documents', 0)} documents with {result.get('chunks', 0)} chunks"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat(query, top_k=5):
    """Chat with the RAG system"""
    global rag_engine, is_indexed

    if not is_indexed:
        return "⚠️  Please index documents first using the Ingest tab"

    if not query:
        return ""

    try:
        # Retrieve relevant context
        results = rag_engine.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant context found. Try a different query."

        # Build context
        context = "\n\n---\n\n".join([
            f"**{r.get('metadata', {}).get('source', 'Unknown')}**\n{r['content'][:500]}"
            for r in results[:3]
        ])

        # Generate answer
        answer = rag_engine.generate(query, [r['content'] for r in results])

        return f"### 📚 Relevant Context\n\n{context}\n\n---\n\n### 💬 Answer\n\n{answer}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def search_codebase(query, top_k=3):
    """Search without generating (just retrieve)"""
    global rag_engine, is_indexed

    if not is_indexed:
        return "⚠️  Please index documents first"

    try:
        results = rag_engine.retrieve(query, top_k=top_k)

        if not results:
            return "No results found"

        output = f"### 🔍 Search Results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"**{i}. {r.get('metadata', {}).get('source', 'Unknown')}**\n"
            output += f"```\n{r['content'][:300]}...\n```\n\n"

        return output

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(title="AI RAG System", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📚 AI RAG System")
    gr.Markdown("Production-ready Retrieval Augmented Generation with chat interface")

    with gr.Tab("💬 Chat"):
        gr.Markdown("### Ask questions about your documents")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K Results")

        def respond(message, history, top_k):
            response = chat(message, top_k)
            history.append((message, response))
            return "", history

        submit_btn.click(respond, [msg, chatbot, top_k_slider], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, top_k_slider], [msg, chatbot])
        clear_btn.click(lambda: (None, []), outputs=[msg, chatbot])

    with gr.Tab("📂 Ingest Documents"):
        gr.Markdown("### Index your documents for searching")

        with gr.Row():
            with gr.Column():
                ingest_path = gr.Textbox(
                    label="Directory or File Path",
                    placeholder="./docs or ./document.pdf"
                )
                chunk_size = gr.Slider(100, 2000, value=1000, step=100, label="Chunk Size")
                embedding = gr.Dropdown(
                    ["openai", "huggingface"],
                    value="openai",
                    label="Embedding Model"
                )
                ingest_btn = gr.Button("📄 Ingest Documents", variant="primary")

            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=5)

        ingest_btn.click(
            initialize_rag,
            inputs=[ingest_path, chunk_size, embedding],
            outputs=status_output
        )

    with gr.Tab("🔍 Search"):
        gr.Markdown("### Search documents (no generation)")

        with gr.Row():
            with gr.Column(scale=3):
                search_query = gr.Textbox(placeholder="Enter search query...")
                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Markdown()

            with gr.Column(scale=1):
                search_top_k = gr.Slider(1, 10, value=3, step=1, label="Top K")

        search_btn.click(
            search_codebase,
            inputs=[search_query, search_top_k],
            outputs=search_output
        )

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## AI RAG System

        A production-ready Retrieval Augmented Generation system that helps you:
        - 📄 Parse documents (PDF, DOCX, TXT, MD)
        - 📊 Smart text chunking
        - 💾 Vector store (ChromaDB, Pinecone, Weaviate)
        - 🔍 Semantic search
        - 💬 Chat with your documents

        ### Requirements
        - Python 3.10+
        - langchain
        - chromadb
        - openai (or huggingface embeddings)
        """)

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting AI RAG System Web UI...")
    print("   Open http://localhost:7860 in your browser")
    app.launch(server_name="0.0.0.0", server_port=7860)