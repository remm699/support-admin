#!/usr/bin/env python3
"""
Index RAG pour Support Admin - CISP
Usage:
  python3 index_docs.py          # indexer
  python3 index_docs.py --rebuild # rebuild complet
"""
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.text_splitter import TokenTextSplitter
import hashlib
import sys
from pathlib import Path

DOCS_DIR       = Path(__file__).parent / "txt"
VECTORSTORE_DIR = Path(__file__).parent / "chroma_db"
COLLECTION     = "cisp-reglementation"
EMBED_MODEL    = "mxbai-embed-large"
CHUNK_SIZE     = 256
CHUNK_OVERLAP  = 32

embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434", embed_batch_size=1)

if "--rebuild" in sys.argv:
    import shutil
    shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
    print("[!] Collection supprimee")

VECTORSTORE_DIR.mkdir(exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = chroma_client.get_or_create_collection(COLLECTION)
print(f"[i] Collection: {collection.count()} vectors")

files = list(DOCS_DIR.glob("*.txt"))
if not files:
    print("[!] Aucun .txt dans txt/")
    sys.exit(1)

text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_nodes = []
for fpath in sorted(files):
    text = fpath.read_text(encoding="utf-8")
    chunks = text_splitter.split_text(text)
    print(f"  {fpath.stem}: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        h = hashlib.sha256(f"{fpath.stem}:{i}:{chunk[:50]}".encode()).hexdigest()[:16]
        all_nodes.append(TextNode(
            id_=f"{fpath.stem}_{i}_{h}",
            text=chunk,
            metadata={"source": fpath.stem, "chunk": i, "file": fpath.name},
        ))

print(f"\n[i] Total: {len(all_nodes)} chunks")

print("[i] Embedding + stockage Chroma...")
for i in range(0, len(all_nodes), 20):
    batch = all_nodes[i:i+20]
    texts = [n.text for n in batch]
    try:
        embs = embed.get_text_embedding_batch(texts)
        collection.add(
            ids=[n.id_ for n in batch],
            embeddings=embs,
            documents=texts,
            metadatas=[n.metadata for n in batch],
        )
        if i % 100 == 0:
            print(f"  {i}-{min(i+20, len(all_nodes))} / {len(all_nodes)}")
    except Exception as e:
        print(f"  [!] batch {i}: {e}")
        for n, t in zip(batch, texts):
            try:
                e_ = embed.get_text_embedding(t)
                collection.add(ids=[n.id_], embeddings=[e_], documents=[t], metadatas=[n.metadata])
            except Exception as e2:
                print(f"    [!] {n.id_}: {e2}")

print(f"[+] Chroma: {collection.count()} vectors")

# Index in-memory
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed)

# Test
print("\n[i] Test retrieval:")
retriever = index.as_retriever(similarity_top_k=3)
results = retriever.retrieve("conditions d'agrement d'un centre CISP")
for r in results:
    print(f"  [{r.score:.3f}] {r.metadata['source']} ch.{r.metadata['chunk']}: {r.text[:100]}...")

# Sauvegarde
import pickle
idx_path = Path(__file__).parent / "index.pkl"
with open(idx_path, "wb") as f:
    pickle.dump({"index": index, "collection_name": COLLECTION, "db_dir": str(VECTORSTORE_DIR)}, f)
print(f"\n[OK] Index: {idx_path}")
print(f"[OK] Chroma: {VECTORSTORE_DIR}")
