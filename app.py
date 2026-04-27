"""
Support Admin — Interface Web RAG
Usage:
  python3 app.py                    # développement (http://localhost:8765)
  uvicorn app:app --host 0.0.0.0 --port 8765 --reload  # production
"""
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(__file__).parent
VECTORSTORE_DIR  = PROJECT_DIR / "chroma_db"
COLLECTION       = "cisp-reglementation"
EMBED_MODEL      = "mxbai-embed-large"
TOP_K            = 5
CUTOFF           = 0.40
PORT             = 8765

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Support Admin — RAG CISP")

# Lazy-loaded index
_index = None

def get_index():
    global _index
    if _index is None:
        embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434", embed_batch_size=1)
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        collection = chroma_client.get_or_create_collection(COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        _index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed)
    return _index

def search_rag(query: str, top_k: int = TOP_K, cutoff: float = CUTOFF):
    index = get_index()
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, embed_model=index._embed_model)
    return [r for r in retriever.retrieve(query) if r.score >= cutoff]

# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = PROJECT_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)

@app.post("/api/search")
async def api_search(q: str = Form(...)):
    if not q.strip():
        return JSONResponse({"results": [], "query": ""})

    try:
        results = search_rag(q)
    except Exception as e:
        return JSONResponse({"results": [], "query": q, "error": str(e)}, status_code=500)

    return JSONResponse({
        "query": q,
        "results": [
            {
                "score": round(r.score, 3),
                "source": r.metadata.get("source", ""),
                "chunk": r.metadata.get("chunk", ""),
                "text": r.text[:600],
            }
            for r in results
        ]
    })

@app.get("/api/status")
async def status():
    try:
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        collection = chroma_client.get_or_create_collection(COLLECTION)
        return JSONResponse({"status": "ok", "vectors": collection.count()})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
