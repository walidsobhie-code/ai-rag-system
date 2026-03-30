#!/usr/bin/env python3
"""
AI RAG System - Using Ollama (free local AI)
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import requests

class RAGEngine:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.docs = []
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder"
    
    def ingest(self, file_path: str) -> Dict:
        path = Path(file_path)
        if not path.exists():
            return {"status": "file_not_found"}
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        self.docs.append({"source": path.name, "content": content[:5000]})
        
        return {"status": "success", "chunks": 1, "file": path.name}
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        # Simple keyword matching for now
        results = []
        for doc in self.docs:
            if any(word.lower() in doc["content"].lower() for word in query.split()[:3]):
                results.append({"text": doc["content"][:500], "score": 0.8, "source": doc["source"]})
        return results[:top_k]
    
    def generate(self, query: str) -> str:
        # Use Ollama for generation
        try:
            context = " ".join([d["content"][:500] for d in self.docs[:3]])
            
            prompt = f"""Based on this context:
{context}

Question: {query}

Answer:"""
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response from AI")
            else:
                return f"AI Error: {response.status_code}"
                
        except Exception as e:
            return f"Ollama not running? Start with: ollama serve\nError: {str(e)}"
    
    def chat(self, query: str) -> Dict:
        answer = self.generate(query)
        sources = self.retrieve(query)
        
        return {
            "answer": answer,
            "sources": sources
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", help="Document to ingest")
    parser.add_argument("--query", help="Query")
    args = parser.parse_args()
    
    rag = RAGEngine()
    
    if args.ingest:
        print(f"📄 Ingesting: {args.ingest}")
        result = rag.ingest(args.ingest)
        print(json.dumps(result, indent=2))
    
    if args.query:
        print(f"\n💬 Question: {args.query}")
        result = rag.chat(args.query)
        print(f"\nAI: {result['answer'][:500]}")

if __name__ == "__main__":
    main()
