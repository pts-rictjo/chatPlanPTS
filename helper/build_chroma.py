#!/usr/bin/env python3
"""
Production-grade build_chroma för juridiska dokument / spektrumdata.
Optimerad för hög retrieval-kvalitet i externa UI/RAG-system.
"""

import os
import json
import csv
import re
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from ollama import Client
from pypdf import PdfReader

# Optional: tabula
try:
    import tabula
    USE_TABULA = True
except ImportError:
    USE_TABULA = False
    print("INFO: tabula-py inte installerad.")

# ==========================
# KONFIG
# ==========================
DATA_ROOT       = Path(os.getenv("DATA_ROOT", "./data"))
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "spectrum_data")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")

MAX_CHARS = 800
OVERLAP = 150
BATCH_SIZE = 64

# ==========================
# INIT
# ==========================
client          = Client(host=OLLAMA_HOST)
# chroma          = chromadb.Client(Settings(persist_directory=CHROMA_DIR)) # LEGACY
chroma          = chromadb.PersistentClient(path=CHROMA_DIR)
collection      = chroma.get_or_create_collection(name=COLLECTION_NAME)

# ==========================
# SEMANTISK CHUNKING
# ==========================
def semantic_split(text):
    """Splitta på juridiska strukturer"""
    parts = re.split(r'(§\s*\d+|Kapitel\s+\d+|Artikel\s+\d+)', text)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) < MAX_CHARS:
            current += part
        else:
            if current.strip():
                chunks.append(current.strip())
            current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def apply_overlap(chunks):
    out = []
    for i, ch in enumerate(chunks):
        if i == 0:
            out.append(ch)
        else:
            prev = chunks[i-1]
            out.append(prev[-OVERLAP:] + ch)
    return out


def chunk_text(text):
    chunks = semantic_split(text)

    # fallback om chunk för stor
    final = []
    for ch in chunks:
        if len(ch) <= MAX_CHARS:
            final.append(ch)
        else:
            i = 0
            while i < len(ch):
                final.append(ch[i:i+MAX_CHARS])
                i += MAX_CHARS - OVERLAP

    return apply_overlap(final)

# ==========================
# TITLE EXTRACTION
# ==========================
def extract_title(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.strip()) > 10:
            return line[:120]
    return text[:120]

# ==========================
# JSON FLATTEN
# ==========================
def flatten_json(data, parent_key=''):
    items = {}

    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)

    elif isinstance(data, list):
        for i, v in enumerate(data):
            items.update(flatten_json(v, f"{parent_key}[{i}]"))

    return items

# ==========================
# TABLE → SEMANTIC TEXT
# ==========================
def table_to_sentences(df):
    rows = []
    cols = list(df.columns)
    for _, r in df.iterrows():
        row = ", ".join(f"{col}: {r[col]}" for col in cols)
        rows.append(row)
    return rows

# ==========================
# EXTRACTORS
# ==========================
def extract_from_pdf(file_path: Path):
    reader = PdfReader(str(file_path))
    items = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            items.append((text, {"type": "pdf_text", "page": page_num}))

    if USE_TABULA:
        try:
            tables = tabula.read_pdf(str(file_path), pages='all', multiple_tables=True)
            for i, table in enumerate(tables):
                if table is not None and not table.empty:
                    for row in table_to_sentences(table):
                        items.append((row, {"type": "pdf_table", "table_index": i}))
        except Exception as e:
            print(f"Tabula-fel: {e}")

    return items


def extract_from_csv(file_path: Path):
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [
            (", ".join(f"{k}: {v}" for k, v in row.items()),
             {"type": "csv_row", "row": i})
            for i, row in enumerate(reader)
        ]


def extract_from_json(file_path: Path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    flat = flatten_json(data)
    return [
        (f"{k}: {v}", {"type": "json_field", "key": k})
        for k, v in flat.items()
    ]


def extract_text(file: Path):
    if file.suffix == ".pdf":
        return extract_from_pdf(file)
    elif file.suffix == ".csv":
        return extract_from_csv(file)
    elif file.suffix == ".json":
        return extract_from_json(file)
    return []

# ==========================
# EMBEDDING (BATCH)
# ==========================
def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        try:
            resp = client.embed(model=EMBED_MODEL, input=batch)
            embeddings.extend(resp["embeddings"])
        except Exception as e:
            print(f"Embedding-fel: {e}")
            embeddings.extend([[0.0]*1024] * len(batch))
    return embeddings

# ==========================
# ID
# ==========================
def make_id(text):
    return hashlib.sha1(text.encode()).hexdigest()

# ==========================
# ENRICHMENT
# ==========================
def enrich(text, meta, file):
    title = extract_title(text)

    enriched = f"""
Titel: {title}
Källa: {file.name}
Typ: {meta.get("type")}
Sida: {meta.get("page", "N/A")}

Innehåll:
{text}
"""

    # BGE instruction prefix
    return f"Represent this document for retrieval:\n{enriched}"

# ==========================
# MAIN
# ==========================
def main():
    if not DATA_ROOT.exists():
        print("DATA_ROOT saknas")
        return

    docs, metas, ids = [], [], []
    seen = set()

    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue

        print(f"Processar {file.name}")
        items = extract_text(file)

        for text, meta in items:
            chunks = chunk_text(text)

            for i, ch in enumerate(chunks):
                ch = ch.strip()
                if not ch:
                    continue

                enriched = enrich(ch, meta, file)

                if enriched in seen:
                    continue
                seen.add(enriched)

                meta_full = {
                    "file": file.name,
                    "file_stem": file.stem,
                    "file_type": file.suffix,
                    "chunk": i,
                    "type": meta.get("type"),
                    "page": meta.get("page", -1),
                }

                docs.append(enriched)
                metas.append(meta_full)
                ids.append(make_id(enriched))

    print(f"Totalt chunks: {len(docs)}")
    if not docs:
        return

    print("Skapar embeddings...")
    embeddings = embed_texts(docs)

    print("Skriver till Chroma...")
    for i in range(0, len(docs), 500):
        collection.add(
            documents=docs[i:i+500],
            embeddings=embeddings[i:i+500],
            metadatas=metas[i:i+500],
            ids=ids[i:i+500]
        )

    print(f"KLAR: {collection.count()} dokument")


if __name__ == "__main__":
    """
    # Exempel på hur du kör:
    # export DATA_ROOT="./data"
    # export CHROMA_DIR="./chroma_db"
    # export OLLAMA_HOST="http://127.0.0.1:11434"
    # python build_chroma.py

    # FÖR ATT LADDA IN EXTERNT
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(persist_directory="./chroma"))
    # OM NYARE CHROMA
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("spectrum_data")
    """
    main()
