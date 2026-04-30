"""
Confidence scoring + Escalade pour Support Admin — CISP
Décide si la réponse du RAG est assez fiable ou s'il faut escalader vers un humain.
"""
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
import requests

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_BASE       = "http://localhost:11434"
LLM_MODEL        = "qwen2.5:0.5b"
PERSONNE_RESSOURCE = "rem"  # TODO: remplacer par vrai contact (Signal/email)

# ── Seuils de confiance ────────────────────────────────────────────────────
class ConfidenceLevel(Enum):
    HIGH   = "high"    # → Réponse directe
    MEDIUM = "medium"  # → Réponse + mention "vérifié par humain"
    LOW    = "low"     # → Escalade humain
    NONE   = "none"    # → Escalade obligatoire

# Seuils sur le nombre de chunks récupérés
CHUNK_COUNT_THRESHOLDS = {
    ConfidenceLevel.HIGH:   3,   # 3+ chunks → haute confiance
    ConfidenceLevel.MEDIUM:  2,   # 2 chunks → confiance moyenne
    ConfidenceLevel.LOW:     1,   # 1 chunk  → faible → escalade
    ConfidenceLevel.NONE:    0,   # 0 chunk  → impossible
}

# Seuil minimum de similarité par chunk
MIN_CHUNK_SIMILARITY = 0.40

# ── Phrases qui indiquent un refus / "je ne sais pas" du LLM ────────────────
LLM_DONT_KNOW_PATTERNS = [
    "je ne peux pas répondre",
    "je ne suis pas en mesure",
    "aucun document",
    "information ne figure pas",
    "pas d'information",
    "je ne trouve pas",
    "je ne connais pas",
    "je n'ai pas",
    "informations disponibles",
    "aucune source",
    "je ne dispose pas",
    "pas trouvé dans les documents",
    "ne figure pas dans les sources",
    "ne provient pas des documents",
    "aucun élément",
    "information insuffisante",
    "je ne suis pas qualifié",
    "hors de mon alcance",
    "je ne suis pas en mesure de",
    "aucun texte",
    "je n'ai aucune information",
]


# ── Score de confiance ──────────────────────────────────────────────────────
def compute_confidence(chunks: list) -> tuple[ConfidenceLevel, str]:
    """
    Calcule le niveau de confiance basé sur les chunks检索.
    
    Args:
        chunks: liste de dicts avec clés 'score' (float), 'text' (str)
    
    Returns:
        (ConfidenceLevel, reason)
    """
    if not chunks:
        return (
            ConfidenceLevel.NONE,
            "Aucun chunk检索é → impossible de répondre"
        )

    # Filtre : on ne garde que les chunks au-dessus du seuil minimum
    quality_chunks = [c for c in chunks if c.get("score", 0) >= MIN_CHUNK_SIMILARITY]
    count = len(quality_chunks)

    if count == 0:
        return (
            ConfidenceLevel.NONE,
            f"Aucun chunk au-dessus du seuil {MIN_CHUNK_SIMILARITY}"
        )

    max_sim   = max(c["score"] for c in quality_chunks)
    avg_sim   = sum(c["score"] for c in quality_chunks) / count

    # Score composite
    # → 3+ chunks ET bonne similarité → HIGH
    # → 2 chunks OU similarité moyenne → MEDIUM
    # → 1 chunk → LOW
    if count >= 3 and avg_sim >= 0.50:
        return (
            ConfidenceLevel.HIGH,
            f"{count} chunks, avg_sim={avg_sim:.3f}, max={max_sim:.3f}"
        )
    elif count >= 2 and avg_sim >= 0.45:
        return (
            ConfidenceLevel.MEDIUM,
            f"{count} chunks, avg_sim={avg_sim:.3f}, max={max_sim:.3f}"
        )
    elif count >= 1:
        return (
            ConfidenceLevel.LOW,
            f"{count} chunk(s), avg_sim={avg_sim:.3f}, max={max_sim:.3f}"
        )
    else:
        return (
            ConfidenceLevel.NONE,
            f"Aucun chunk au-dessus du seuil {MIN_CHUNK_SIMILARITY}"
        )


def llm_says_dont_know(answer: str) -> bool:
    """Détecte si le LLM a refusé de répondre."""
    text = answer.lower()
    return any(pat in text for pat in LLM_DONT_KNOW_PATTERNS)


# ── Escalade ────────────────────────────────────────────────────────────────
ESCALADE_DIR = Path("/home/rem/projets/support-admin/escalades")
ESCALADE_DIR.mkdir(exist_ok=True)


def escalate(question: str, chunks: list, answer: str, confidence: ConfidenceLevel, reason: str):
    """
    Génère une escalade : sauvegarde + notification personne ressource.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ts   = timestamp  # pour le nom de fichier
    
    escalation = {
        "timestamp":     datetime.now().isoformat(),
        "question":      question,
        "confidence":    confidence.value,
        "reason":        reason,
        "chunks_count": len(chunks),
        "chunks": [
            {
                "score":  c["score"],
                "source": c["source"],
                "chunk":  c["chunk"],
                "text":   c["text"][:300],
            }
            for c in chunks
        ],
        "llm_answer": answer[:2000],
        "status": "pending",
    }

    # ── 1. Sauvegarde sur disque ──────────────────────────────────────────
    filepath = ESCALADE_DIR / f"escalade_{safe_ts}.json"
    filepath.write_text(json.dumps(escalation, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 2. Notification personne ressource ────────────────────────────────
    # TODO: brancher sur Signal / email / etc.
    # Pour l'instant: simple notification via curl vers un webhook
    _notify_personne_ressource(escalation, str(filepath))

    return str(filepath)


def _notify_personne_ressource(escalation: dict, filepath: str):
    """
    Envoie une notification à la personne ressource.
    Actuellement: fichier local .json + log.
    À brancher sur Signal / email / webhook / Telegram.
    """
    question = escalation["question"]
    reason   = escalation["reason"]
    
    # Log pour debug / surveillance
    print(f"\n[ESCALADE] → {PERSONNE_RESSOURCE}")
    print(f"  Question : {question[:100]}")
    print(f"  Raison   : {reason}")
    print(f"  Fichier  : {filepath}\n")

    # ── Brancher ici ta méthode de notification favorite : ──────────────
    # Signal : signal-cli send …
    # Email  : curl SMTP …
    # Webhook: requests.post("https://...", json=escalation)
    # Telegram: bot.send_message(chat_id=..., text=...)
    #
    # Exemple webhook (Mattermost/Slack):
    # try:
    #     requests.post(
    #         "https://hooks.slack.com/...",
    #         json={"text": f"⚠️ Escalade RAG\nQuestion: {question}\nRaison: {reason}"},
    #         timeout=10,
    #     )
    # except Exception as e:
    #     print(f"[Webhook failed] {e}")


# ── Réponse d'attente pour l'utilisateur ─────────────────────────────────
ESCALADE_USER_MESSAGE = (
    "Votre question a bien été reçue.\n\n"
    "Le système n'a pas pu y répondre de manière suffisamment fiable "
    "avec les documents disponibles.\n\n"
    "→ Votre demande a été transmise à notre personne ressource CISP, "
    "qui vous recontactera sous peu.\n\n"
    "Merci de votre patience."
)

MEDIUM_USER_MESSAGE = (
    "Réponse trouvée, mais avec une confiance limitée.\n\n"
    "⚠️ Merci de vérifier cette information auprès de votre personne ressource CISP "
    "avant de prendre une décision.\n\n"
    "Si l'information vous semble incomplète, vous pouvez reformuler votre question."
)


def get_user_message(confidence: ConfidenceLevel) -> str:
    if confidence == ConfidenceLevel.LOW:
        return ESCALADE_USER_MESSAGE
    elif confidence == ConfidenceLevel.MEDIUM:
        return MEDIUM_USER_MESSAGE
    return ""  # HIGH → pas de message particulier


if __name__ == "__main__":
    # Test rapide en CLI
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 confidence.py \"votre réponse LLM\"")
        sys.exit(1)

    answer = " ".join(sys.argv[1:])
    dont_know = llm_says_dont_know(answer)
    print(f"[Answer] {answer}")
    print(f"[DontKnow] {dont_know}")
