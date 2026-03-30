# AI RAG System 📚🤖

Production-ready Retrieval Augmented Generation system with knowledge base. Build AI that knows your documents.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

## Why RAG?

Every company is building AI that knows their data. RAG is the foundation of enterprise AI.

## ✨ Features

- 📄 **Document Parsing** - PDF, DOCX, TXT, MD support
- 📊 **Smart Chunking** - Multiple chunking strategies
- 💾 **Vector Store** - ChromaDB, Pinecone, Weaviate support
- 🔍 **Semantic Search** - Find relevant context
- 💬 **Chat UI** - Gradio chat interface
- 🧠 **Multi-LLM** - OpenAI, Anthropic, local models
- 🌐 **Web UI** - Beautiful Gradio interface
- 🐳 **Docker Ready** - Deploy anywhere

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Index documents
python rag_engine.py --ingest ./docs

# Start chat UI
python gradio_app.py
# Open http://localhost:7860
```

## 🐳 Docker

```bash
docker build -t ai-rag-system .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key ai-rag-system
```

## 📝 License

MIT License

## ⭐ Star if helpful!