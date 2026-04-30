"""
Support Admin — Interface Web RAG
Usage:
  python3 app.py                    # développement (http://localhost:8765)
  uvicorn app:app --host 0.0.0.0 --port 8765 --reload  # production
"""
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import chromadb, requests
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from pathlib import Path

from router import route, RouteResult, get_refusal_message
from confidence import (
    compute_confidence,
    llm_says_dont_know,
    escalate,
    get_user_message,
    ConfidenceLevel,
)
import re

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(__file__).parent
VECTORSTORE_DIR  = PROJECT_DIR / "chroma_db"
COLLECTION       = "cisp-reglementation"
EMBED_MODEL     = "mxbai-embed-large"
LLM_MODEL       = "qwen2.5:0.5b"
OLLAMA_BASE     = "http://localhost:11434"
TOP_K           = 5
CUTOFF          = 0.40
PORT            = 8765

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

def validate_answer(answer: str, chunks: list) -> tuple[bool, str]:
    """
    Vérifie que les claims numériques de la réponse sont présents dans les chunks.
    Retourne (is_valid, reason).
    """
    if not chunks or not answer:
        return True, ""

    # Patterns de claims potentiellement hallucinations :
    # - Montants (e.g. "347.4 millions", "10.000 euros")
    # - Pourcentages précis (e.g. "10%")
    # - Dates spécifiques (e.g. "2019", "2025")
    # - Nombres de personnes
    number_patterns = [
        r'(\d+\.?\d*)\s*(millions?|milliard?|euros?|€)',
        r'(\d+\.?\d*)\s*%',
        r'(\d{4})-\d{2}-\d{2}',
        r"(\d{1,3}(?:\.\d{3})+)",
    ]

    all_chunk_text = " ".join(c["text"].lower() for c in chunks)

    for pattern in number_patterns:
        for match in re.finditer(pattern, answer.lower()):
            number = match.group(1).replace(".", "")
            if number not in all_chunk_text:
                # Vérification plus large : le nombre apparaît-il sous une forme proche ?
                found = False
                for chunk in chunks:
                    if number in chunk["text"].lower():
                        found = True
                        break
                if not found:
                    return False, f"Claim non sourcée dans les chunks : '{match.group(0)}'"

    return True, ""


def synthesize(query: str, chunks: list) -> str:
    """Appelle qwen2.5:0.5b pour synthétiser une réponse à partir des chunks."""
    if not chunks:
        return "Aucun document pertinent trouvé pour répondre à cette question."

    context = "\n\n".join(
        f"[Source {i+1}] {c['source']} (chunk {c['chunk']}):\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""Tu es un assistant spécialisé en administration des Centres d'Insertion Socioprofessionnelle (CISP) en Région wallonne de Belgique.

Tu répond EXCLUSIVEMENT à partir des sources fournies ci-dessous. Tu ne dois JAMAIS inventer, amplifier ou compléter une information qui ne figure pas littéralement dans les sources.

RÈGLES ABSOLUES :
1. Ne cite que des informations présentes mot pour mot (ou très proche) dans les sources
2. Si les sources ne contiennent pas l'information pour répondre à la question, réponds UNIQUEMENT : "Je ne dispose pas d'assez d'informations dans les sources pour répondre à cette question."
3. Ne mentionne pas de montants, dates, chiffres, noms s'ils ne figurent pas explicitement dans les sources
4. Pour chaque affirmation, indique quelle source (numéro) la supporte

---

Question: {query}

---

Sources:
{context}

---

Réponse (en français, maximum 300 mots,结构和 claire) :"""

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[Erreur de synthèse LLM: {e}]"

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

    # ── 1. Guardrail : router check ──────────────────────────────────────
    route_result, route_reason = route(q)
    if route_result == RouteResult.OUT_OF_TOPIC:
        return JSONResponse({
            "query": q,
            "answer": get_refusal_message(),
            "results": [],
            "router": {"decision": route_result, "reason": route_reason},
            "confidence": None,
        })

    # ── 2. RAG retrieval ──────────────────────────────────────────────────
    try:
        results = search_rag(q)
    except Exception as e:
        return JSONResponse({"results": [], "query": q, "error": str(e)}, status_code=500)

    chunks = [
        {
            "score": r.score,
            "source": r.metadata.get("source", ""),
            "chunk": r.metadata.get("chunk", ""),
            "text": r.text[:600],
        }
        for r in results
    ]

    # ── 3. Confidence scoring ─────────────────────────────────────────────
    conf_level, conf_reason = compute_confidence(chunks)

    # ── 4. Escalade si confiance trop basse ──────────────────────────────
    if conf_level in (ConfidenceLevel.NONE, ConfidenceLevel.LOW):
        filepath = escalate(q, chunks, "", conf_level, conf_reason)
        return JSONResponse({
            "query": q,
            "answer": get_user_message(conf_level),
            "results": chunks,
            "router": {"decision": route_result, "reason": route_reason},
            "confidence": {
                "level": conf_level.value,
                "reason": conf_reason,
                "escalated": True,
                "file": filepath,
            },
        })

    # ── 5. Synthèse LLM ──────────────────────────────────────────────────
    answer = synthesize(q, chunks)

    # ── 5b. Validation des claims numériques ────────────────────────────
    is_valid, validation_reason = validate_answer(answer, chunks)

    # ── 6. Détection "je ne sais pas" → escalade même si confiance OK ────
    if llm_says_dont_know(answer):
        filepath = escalate(q, chunks, answer, ConfidenceLevel.LOW, "LLM says don't know")
        return JSONResponse({
            "query": q,
            "answer": get_user_message(ConfidenceLevel.LOW),
            "results": chunks,
            "router": {"decision": route_result, "reason": route_reason},
            "confidence": {
                "level": conf_level.value,
                "reason": f"{conf_reason} — LLM a refusé de répondre",
                "escalated": True,
                "file": filepath,
            },
        })

    # ── 6b. Claim non sourcé → escalade ────────────────────────────────
    if not is_valid:
        filepath = escalate(q, chunks, answer, ConfidenceLevel.LOW, f"Hallucination détectée: {validation_reason}")
        return JSONResponse({
            "query": q,
            "answer": get_user_message(ConfidenceLevel.LOW),
            "results": chunks,
            "router": {"decision": route_result, "reason": route_reason},
            "confidence": {
                "level": conf_level.value,
                "reason": f"{conf_reason} — {validation_reason}",
                "escalated": True,
                "file": filepath,
            },
        })

    # ── 7. Réponse normale (HIGH ou MEDIUM) ──────────────────────────────
    user_message = get_user_message(conf_level)
    final_answer = (user_message + "\n\n" + answer).strip() if user_message else answer

    return JSONResponse({
        "query": q,
        "answer": final_answer,
        "results": chunks,
        "router": {"decision": route_result, "reason": route_reason},
        "confidence": {
            "level": conf_level.value,
            "reason": conf_reason,
            "escalated": False,
        },
    })

@app.get("/api/status")
async def status():
    try:
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        collection = chroma_client.get_or_create_collection(COLLECTION)
        return JSONResponse({"status": "ok", "vectors": collection.count()})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/api/route")
async def api_route(q: str = Form(...)):
    """Teste le router seul, sans RAG."""
    if not q.strip():
        return JSONResponse({"query": ""})
    result, reason = route(q)
    return JSONResponse({
        "query": q,
        "decision": result,
        "reason": reason,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
