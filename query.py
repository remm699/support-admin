#!/usr/bin/env python3
"""
Query engine RAG pour Support Admin - CISP
Usage:
  python3 query.py "votre question"
"""
import sys
import pickle
from pathlib import Path
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
import chromadb

VECTORSTORE_DIR = Path(__file__).parent / "chroma_db"
COLLECTION     = "cisp-reglementation"
EMBED_MODEL    = "mxbai-embed-large"
LLM_MODEL      = "qwen2.5:7b"
TOP_K          = 5
SIMILARITY_CUTOFF = 0.45

def load_index():
    embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434", embed_batch_size=1)
    chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = chroma_client.get_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed)
    return index, embed

def query(question: str, top_k=TOP_K, similarity_cutoff=SIMILARITY_CUTOFF):
    index, embed = load_index()
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, embed_model=embed)
    results = retriever.retrieve(question)
    
    # Filter by similarity
    relevant = [r for r in results if r.score >= similarity_cutoff]
    return relevant

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 query.py \"votre question\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    print(f"[?] Question: {question}\n")
    
    results = query(question)
    
    if not results:
        print("[!] Aucun resultat pertinent trouve. Reformulez la question ou elargissez le seuil de similarite.")
        sys.exit(0)
    
    print(f"[+] {len(results)} resultats trouves:\n")
    for i, r in enumerate(results, 1):
        print(f"--- Resultat {i} [{r.score:.3f}] ---")
        print(f"Source: {r.metadata['source']} (chunk {r.metadata['chunk']})")
        print(f"Texte: {r.text[:500]}")
        print()
