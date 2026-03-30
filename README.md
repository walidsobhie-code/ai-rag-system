# AI RAG System 📚🤖

Production-ready Retrieval Augmented Generation system with knowledge base. Build AI that knows your documents.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Why RAG?

Every company is building AI that knows their data. RAG is the foundation of enterprise AI.

## ✨ Features

- 📄 **Document Parsing** - PDF, DOCX, TXT, MD support
- 📊 **Smart Chunking** - Multiple chunking strategies
- 💾 **Vector Store** - ChromaDB, Pinecone, Weaviate support
- 🔍 **Semantic Search** - Find relevant context
- 💬 **Chat UI** - Gradio chat interface
- 🧠 **Multi-LLM** - OpenAI, Anthropic, local models

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Index documents
python rag_engine.py --ingest ./docs

# Start chat
python chat.py
```

## 📖 Documentation

- [Setup Guide](docs/setup.md)
- [Configuration](docs/config.md)
- [Examples](examples/)

## 🛠️ Requirements

```
langchain>=0.1.0
chromadb>=0.4.0
pypdf>=3.0.0
openai>=1.0.0
gradio>=4.0.0
```

## 📝 License

MIT License

## ⭐ Star if helpful!
