"""Tests for AI RAG System"""
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_rag_engine_import():
    """Test rag_engine module imports"""
    from rag_engine import RAGEngine, DocumentLoader, TextChunker
    assert RAGEngine is not None


def test_rag_engine_init():
    """Test RAGEngine initialization"""
    from rag_engine import RAGEngine
    rag = RAGEngine()
    assert rag is not None
    assert rag.is_indexed is False


def test_document_loader():
    """Test DocumentLoader can be imported"""
    from rag_engine import DocumentLoader
    assert DocumentLoader is not None


def test_text_chunker():
    """Test TextChunker initialization"""
    from rag_engine import TextChunker
    chunker = TextChunker(chunk_size=500)
    assert chunker is not None
    assert chunker.chunk_size == 500


def test_rag_retrieve_not_indexed():
    """Test retrieve returns empty when not indexed"""
    from rag_engine import RAGEngine
    rag = RAGEngine()
    results = rag.retrieve("test query")
    assert results == []


def test_rag_generate():
    """Test generate method"""
    from rag_engine import RAGEngine
    rag = RAGEngine()
    result = rag.generate("test query")
    assert isinstance(result, str)
    assert len(result) > 0