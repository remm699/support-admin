#!/usr/bin/env python3
"""
Surveillance des nouveaux documents sur le NAS et ré-indexation automatique.
Usage: python3 monitor_docs.py        (mode one-shot: vérifie et indexe si nouveaux fichiers)
       python3 monitor_docs.py --daemon (mode daemon: surveille en continu)

Le script surveille ~/projets/support-admin/txt/ pour les nouveaux fichiers .txt
et déclenche une ré-indexation automatique.
"""
import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
TXT_DIR      = PROJECT_DIR / "txt"
WATCH_DIR    = None  # None = automatique via NAS mount ou local
NAS_DOCS     = "/DATA/Documents/partage/projet/support-admin/documentation"
NAS_HOST     = "192.168.0.49"
NAS_USER     = "rem"

STATE_FILE   = PROJECT_DIR / ".monitor_state"

# ── SSH helper ────────────────────────────────────────────────────────────────
SSH_ASKPASS  = """#!/bin/bash
echo 'Foufette699!'
"""

def ssh_askpass_script(path="/tmp/nas_askpass.sh"):
    Path(path).write_text(SSH_ASKPASS)
    os.chmod(path, 0o700)
    return path

def nas_ls():
    script = ssh_askpass_script()
    env = os.environ.copy()
    env["SSH_ASKPASS"] = script
    env["DISPLAY"] = "dummy"
    env["SSH_ASKPASS_REQUIRE"] = "force"
    result = subprocess.run(
        ["ssh", "-o", "ASKPASS_REQUIRE=force", f"{NAS_USER}@{NAS_HOST}",
         f"ls -la '{NAS_DOCS}/'"],
        capture_output=True, text=True, env=env, timeout=15
    )
    return result.stdout

def nas_fetch_file(fname):
    script = ssh_askpass_script()
    env = os.environ.copy()
    env["SSH_ASKPASS"] = script
    env["DISPLAY"] = "dummy"
    env["SSH_ASKPASS_REQUIRE"] = "force"
    local_path = TXT_DIR / fname.replace(".pdf", ".txt").replace(".docx", ".txt")
    result = subprocess.run(
        ["scp", f"{NAS_USER}@{NAS_HOST}:'{NAS_DOCS}/{fname}'", f"{TXT_DIR}/"],
        capture_output=True, text=True, env=env, timeout=60
    )
    return result.returncode == 0

# ── State management ──────────────────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        return set(STATE_FILE.read_text().splitlines())
    return set()

def save_state(filenames):
    STATE_FILE.write_text("\n".join(filenames))

# ── Extraction texte ─────────────────────────────────────────────────────────
def extract_text(nas_fname):
    script = ssh_askpass_script()
    env = os.environ.copy()
    env["SSH_ASKPASS"] = script
    env["DISPLAY"] = "dummy"
    env["SSH_ASKPASS_REQUIRE"] = "force"
    
    local_pdf = TXT_DIR / nas_fname
    local_txt = TXT_DIR / (nas_fname.rsplit(".", 1)[0] + ".txt")
    
    if nas_fname.lower().endswith(".pdf"):
        result = subprocess.run(
            ["scp", f"{NAS_USER}@{NAS_HOST}:'{NAS_DOCS}/{nas_fname}'", str(local_pdf)],
            capture_output=True, text=True, env=env, timeout=60
        )
        if result.returncode != 0:
            print(f"[!] Erreur copie PDF: {result.stderr}")
            return False
        
        import fitz
        doc = fitz.open(local_pdf)
        text = "".join(page.get_text() for page in doc)
        local_txt.write_text(text, encoding="utf-8")
        local_pdf.unlink(missing_ok=True)
        
    elif nas_fname.lower().endswith(".docx"):
        result = subprocess.run(
            ["scp", f"{NAS_USER}@{NAS_HOST}:'{NAS_DOCS}/{nas_fname}'", str(TXT_DIR / nas_fname)],
            capture_output=True, text=True, env=env, timeout=60
        )
        if result.returncode != 0:
            print(f"[!] Erreur copie DOCX: {result.stderr}")
            return False
        import docx
        doc = docx.Document(TXT_DIR / nas_fname)
        text = "\n".join(p.text for p in doc.paragraphs)
        local_txt.write_text(text, encoding="utf-8")
        (TXT_DIR / nas_fname).unlink(missing_ok=True)
    else:
        return False
    
    print(f"  [+] Extrait: {nas_fname} -> {local_txt.name} ({len(text)} chars)")
    return True

# ── Sync & index ─────────────────────────────────────────────────────────────
def check_and_sync():
    """Vérifie les nouveaux fichiers et les extrait."""
    print("[i] Scan NAS documentation/")
    try:
        output = nas_ls()
    except Exception as e:
        print(f"[!] Erreur connexion NAS: {e}")
        return set()
    
    lines = [l for l in output.splitlines() if l.endswith(".pdf") or l.endswith(".docx") or l.endswith(".PDF") or l.endswith(".DOCX")]
    current_files = {l.split()[-1] for l in lines if len(l.split()) >= 8}
    known_files = load_state()
    new_files = current_files - known_files
    
    if not new_files:
        print("  [=] Aucun nouveau fichier")
        return set()
    
    print(f"  [+] {len(new_files)} nouveau(x) fichier(s): {new_files}")
    for fname in new_files:
        extract_text(fname)
    
    save_state(current_files)
    return new_files

def run_index():
    """Lance l'indexation."""
    print("[i] Lancement index_docs.py...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "index_docs.py"), "--rebuild"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        print("[+] Indexation terminee")
    else:
        print(f"[!] Erreur indexation: {result.stderr[-500:]}")
    return result.returncode == 0

# ── Watchdog handler ─────────────────────────────────────────────────────────
class DocHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.pending = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".txt"):
            print(f"[i] Nouveau txt detecte: {event.src_path}")
            self.pending.add(Path(event.src_path).name)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".txt"):
            self.pending.add(Path(event.src_path).name)

def run_daemon():
    """Surveillance continue du dossier local txt/."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    
    event_handler = DocHandler()
    observer = Observer()
    observer.schedule(event_handler, str(TXT_DIR), recursive=False)
    observer.start()
    print(f"[i] Daemon actif - surveillance {TXT_DIR}")
    
    try:
        while True:
            time.sleep(60)
            if event_handler.pending:
                print(f"[i] {len(event_handler.pending)} fichier(s) en attente: {event_handler.pending}")
                event_handler.pending.clear()
                run_index()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--daemon" in sys.argv:
        run_daemon()
    else:
        # Mode one-shot: vérifie et ré-indexe si besoin
        new = check_and_sync()
        if new:
            run_index()
        else:
            print("[i] Rien a synchroniser")
