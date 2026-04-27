# Support Admin — RAG Administratif CISP

Agent RAG pour la recherche documentaire sur la réglementation wallonne des Centres d'Insertion Socioprofessionnelle (CISP).

## Structure

```
support-admin/
├── index_docs.py      # Script d'indexation RAG
├── query.py           # Script de query/interrogation
├── monitor_docs.py    # Surveillance NAS + ré-index auto
├── .gitignore
├── docs/              # Documents originaux (PDF/DOCX du NAS)
└── txt/               # Texte extrait des documents
```

**Ne pas committer** : `chroma_db/` (vecteurs regeneres par `index_docs.py`), `index.pkl` (reconstruit depuis Chroma).

## Dépendances

```bash
pip install pymupdf python-docx chromadb llama-index llama-index-vector-stores-chroma llama-index-embeddings-ollama watchdog fastapi python-multipart --break-system-packages
```

## Interface Web

```bash
python3 app.py
# → http://localhost:8765
```

Serveur accessible sur `0.0.0.0:8765` (réseau local). Pour y accéder depuis une autre machine :
```
http://<adresse-IP>:8765
```

## Usage

```bash
# Indexer / ré-indexer les documents
python3 index_docs.py --rebuild

# Interroger le RAG
python3 query.py "votre question sur la réglementation CISP"

# Surveillance automatique (cron)
python3 monitor_docs.py
```

## Documents indexés

- Décret CISP consolidé (2020)
- Arrêté AI (2012)
- AGW d'exécution du décret CISP
- AGW относительно des dépenses éligibles
- etc.

## Stack

- **Embeddings**: `mxbai-embed-large` (Ollama)
- **Vector DB**: Chroma (persistant)
- **LLM**: `qwen2.5:7b` (Ollama)
- **Indexation**: llama-index
