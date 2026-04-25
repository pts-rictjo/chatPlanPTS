#!/usr/bin/env python3
import os
import json
import csv
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings
from ollama import Client
from pypdf import PdfReader

# ==========================
# CONFIG
# ==========================

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "/chroma")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "spectrum_data")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# standard val
# EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
#
# större kontextfönster
# nomic och bge-m3 (båda har 8k tokens enligt dokumentation)
# EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")            # lättare och snabbare
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")                        # Byt till bge-m3 för svenska dokument
# EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed:137m") # Ett sista alternativ
#
# Chunks :
#   färre (större) ger mer sammanhang,
#   fler (mindre) ger mer precision.
#       Med bge-m3 rekommenderas MAX_CHARS ≈ 800–1000
MAX_CHARS = 900 # uptill 1200 eller mer med de nya modeller
OVERLAP   = 200 # Justera overlap efter behov
#
# Om det behöver bantas
#MAX_CHARS = 400
#OVERLAP = 80

# ==========================
# INIT
# ==========================

client = Client(host=OLLAMA_HOST)

chroma = chromadb.Client(
    Settings(persist_directory=CHROMA_DIR)
)

collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

# ==========================
# HELPERS
# ==========================

def chunk(text):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+MAX_CHARS])
        i += MAX_CHARS - OVERLAP
    return out

def doc_id(*parts):
    return hashlib.sha1("::".join(parts).encode()).hexdigest()

def embed(texts):
    """Embed a list of strings one by one (Ollama requires a single string per call)."""
    embeddings = []
    for text in texts:
        resp = client.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(resp["embedding"])
    return embeddings

# ==========================
# GENERIC FILE LOADER
# ==========================

def extract_text(file: Path):
    if file.suffix == ".csv":
        with open(file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return "\n".join(str(row) for row in reader)

    elif file.suffix == ".json":
        return json.dumps(json.load(open(file)), ensure_ascii=False)

    elif file.suffix == ".pdf":
        reader = PdfReader(str(file))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    else:
        return None

# ==========================
# MAIN INDEX
# ==========================

def main():
    # Check if data root exists
    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT = '{DATA_ROOT}' does not exist. Set DATA_ROOT environment variable to a folder with .pdf, .csv, or .json files.")
        return

    docs, ids, metas = [], [], []

    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue

        text = extract_text(file)
        if not text:
            continue

        for i, ch in enumerate(chunk(text)):
            docs.append(ch)
            ids.append(doc_id(file.name, str(i)))
            metas.append({"file": file.name})

    if not docs:
        print(f"No documents found in '{DATA_ROOT}'. Supported types: .pdf, .csv, .json")
        return

    print(f"Embedding {len(docs)} chunks...")

    embeddings = embed(docs)

    collection.add(
        documents=docs,
        ids=ids,
        embeddings=embeddings,
        metadatas=metas
    )

    print(f"Successfully added {collection.count()} items to ChromaDB at '{CHROMA_DIR}'.")

if __name__ == "__main__":
    """
    $ export DATA_ROOT="./data"
    $ export CHROMA_DIR="./chroma_db"
    $ export OLLAMA_HOST="http://127.0.0.1:11434"
    $ python build_chroma.py
    OR
    $ OLLAMA_HOST="http://127.0.0.1:11434" python build_chroma.py
    """
    main()
