"""
Router / Guardrail pour Support Admin — CISP
Décide si une question est dans le sujet (administratif CISP) ou non.
"""
import re
import requests

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_BASE   = "http://localhost:11434"
EMBED_MODEL   = "mxbai-embed-large"

# ── Mots-clés acceptés (administratif CISP / Région wallonne) ──────────────

# Level 1 — Mots forts (1 seul match = accept)
STRONG_KEYWORDS = [
    # CISP
    "cisp",
    # Subsides & financement
    "subside", "subsides", "subvention", "subventions", "financement",
    "dépenses éligibles", "titre de dépenses",
    # Réglementation
    "décret-cisp", "décret cisp", "ai-", "agrément", "agrément",
    "arrêté du gouvernement wallon",
    # Versions spécifiques (noms fichiers)
    "01-04-2021", "12.01.2012", "28.06.2012", "29.04.19", "13.02.2020",
]

# Level 2 — Mots faibles (requièrent 2+ matches pour accepter)
WEAK_KEYWORDS = [
    "région wallonne", "wallonnie", "wallon",
    "insertion socioprofessionnelle", "insertion socio-professionnelle",
    "demandeur d'emploi", "demandeurs d'emploi",
    "dispositif", "marché du travail", "chômeur", "chômeurs",
    "bruxelles", "bfp", "forem", "actiris",
    "bruxelles formation", "bruxelles-formation",
    "agw", "gie", "gres", "cisp-gie",
    "règlement", "règlementation", "réglementation", "consolidée",
    "procédure", "procédures", "déclaration", "rapport", "rapports",
    "bilan", "comptes", "contrôle", "audit", "convention",
    "conventions", "conditions d'agrément",
    "formation professionnelle", "formations",
    "formation individualisée", "stagiaire", "stagiaires",
    "formatteur", "formateurs",
    "pièces justificatives", "justificatifs",
    "version consolidée",
]

# Mots qui indiquent clairement HORS SUJET
REJECTED_KEYWORDS = [
    "météo", "sport", "recette", "cuisine", "politique", "actualité",
    "news", "film", "série", "jeu vidéo", "cinéma",
    "sexualité", "religion", "pronostic", "paris",
]

# ── Seuil embedding ─────────────────────────────────────────────────────────
EMBED_SIMILARITY_THRESHOLD = 0.60  # en dessous → hors sujet


def _cosine_similarity(a: list, b: list) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(text: str) -> list:
    """Retourne le vecteur d'embedding via Ollama."""
    r = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _normalize(text: str) -> str:
    """Normalise le texte pour la recherche de keywords."""
    return text.lower()


# ── Exemples d'embedding pour le fallback ──────────────────────────────────
# On calcule ces vecteurs une fois au démarrage (lazy)
_EXAMPLE_QUERIES = [
    "Quelles sont les conditions d'agrément d'un CISP en Région wallonne ?",
    "Comment fare une demande de subside pour un centre d'insertion ?",
    "Quelles dépenses sont éligibles dans le cadre d'un subside CISP ?",
    "Décret CISP et ses modalités d'application",
    "Procédure de remboursement des dépenses pour un CISP",
    "Qui est la personne ressource pour les subsides CISP ?",
]

# Cache des embeddings d'exemples (calculés une fois)
_example_embeddings: list | None = None


def _get_example_embeddings() -> list:
    global _example_embeddings
    if _example_embeddings is None:
        _example_embeddings = [_get_embedding(q) for q in _EXAMPLE_QUERIES]
    return _example_embeddings


# ── Résultat du router ──────────────────────────────────────────────────────
class RouteResult:
    IN_TOPIC    = "in_topic"      # → RAG
    OUT_OF_TOPIC = "out_of_topic"  # → Refuser poliment
    LOW_CONFIDENCE = "low_confidence"  # → Escalade humain


def route(question: str) -> tuple[RouteResult, str]:
    """
    Analyse la question et decide si on la traite ou non.
    
    Returns:
        (RouteResult, reason)
    """
    text = _normalize(question)

    # ── 1. Rejet rapide : mots clairement hors sujet ─────────────────────────
    for kw in REJECTED_KEYWORDS:
        if kw in text:
            return (
                RouteResult.OUT_OF_TOPIC,
                f"Mot détecté hors sujet : '{kw}' — Cette question ne concerne pas l'administration CISP."
            )

    # ── 2. Level 1 — Mots forts (1 seul match = accept) ───────────────────
    strong_matches = [kw for kw in STRONG_KEYWORDS if kw in text]
    if strong_matches:
        return (
            RouteResult.IN_TOPIC,
            f"Keywords forts matchés : {strong_matches}"
        )

    # ── 3. Level 2 — Mots faibles (2+ matches requis) ───────────────────
    weak_matches = [kw for kw in WEAK_KEYWORDS if kw in text]
    if len(weak_matches) >= 2:
        return (
            RouteResult.IN_TOPIC,
            f"Keywords faibles matchés ({len(weak_matches)}) : {weak_matches[:4]}"
        )

    # ── 4. Fallback embedding similarity ──────────────────────────────────
    # Ni rejeté ni accepté par keywords → on compare aux exemples
    try:
        query_emb = _get_embedding(question)
        example_embs = _get_example_embeddings()
        similarities = [
            _cosine_similarity(query_emb, ex_emb)
            for ex_emb in example_embs
        ]
        max_sim = max(similarities) if similarities else 0.0

        if max_sim >= EMBED_SIMILARITY_THRESHOLD:
            return (
                RouteResult.IN_TOPIC,
                f"Similarité embedding {max_sim:.3f} (seuil: {EMBED_SIMILARITY_THRESHOLD})"
            )
        else:
            return (
                RouteResult.OUT_OF_TOPIC,
                f"Similarité embedding {max_sim:.3f} trop basse (seuil: {EMBED_SIMILARITY_THRESHOLD})"
            )
    except Exception as e:
        # Si Ollama est down, on préfère être permissif plutôt que refuser
        return (
            RouteResult.LOW_CONFIDENCE,
            f"Erreur embedding ({e}) — escalate to humain"
        )


# ── Réponse automatique pour refus ──────────────────────────────────────────
REFUSAL_MESSAGE = (
    "Désolé, je suis un assistant spécialisé dans l'administration des "
    "Centres d'Insertion Socioprofessionnelle (CISP) en Région wallonne.\n\n"
    "Je ne peux répondre qu'aux questions concernant :\n"
    "• Les subsides et financements CISP\n"
    "• Les décrets et règlements (décret CISP, AGW...)\n"
    "• Les procédures d'agrément et de reporting\n"
    "• Les dépenses éligibles\n\n"
    "Pour toute autre question, contactez votre personne ressource CISP."
)


def get_refusal_message() -> str:
    return REFUSAL_MESSAGE


if __name__ == "__main__":
    # Test rapide en CLI
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 router.py \"votre question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result, reason = route(question)

    print(f"[Question] {question}")
    print(f"[Route] {result}")
    print(f"[Reason] {reason}")
