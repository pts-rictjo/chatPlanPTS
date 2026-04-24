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
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

MAX_CHARS = 1200
OVERLAP = 200

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
    return client.embeddings(model=EMBED_MODEL, prompt=texts)["embedding"]

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

    print(f"Embedding {len(docs)} chunks...")

    embeddings = embed(docs)

    collection.add(
        documents=docs,
        ids=ids,
        embeddings=embeddings,
        metadatas=metas
    )

    print("KLAR:", collection.count())

if __name__ == "__main__":
    main()
