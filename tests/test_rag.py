#!/usr/bin/env python3
"""Tests for AI RAG System"""
import unittest
import subprocess
import os

class TestRAG(unittest.TestCase):
    
    def test_rag_engine_exists(self):
        self.assertTrue(os.path.exists('rag_engine.py'))
    
    def test_rag_engine_runs(self):
        result = subprocess.run(['python3', 'rag_engine.py'],
                              capture_output=True, text=True, timeout=10)
        self.assertIn('ingesting', result.stdout)

if __name__ == '__main__':
    unittest.main()
