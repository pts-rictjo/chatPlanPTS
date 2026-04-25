#!/usr/bin/env python3

import os
import json
import csv
import re
import hashlib
from pathlib import Path

import chromadb
from chromadb import PersistentClient
from ollama import Client
from pypdf import PdfReader

try:
    import tabula
    USE_TABULA = True
except ImportError:
    USE_TABULA = False

# ==========================
# KONFIG
# ==========================
DATA_ROOT       = Path(os.getenv("DATA_ROOT", "./data"))
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "spectrum_data")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")

MAX_CHARS = 600
MIN_CHARS = 200
OVERLAP   = 80
BATCH_SIZE = 32

# ==========================
# INIT
# ==========================
print(f"Använder ChromaDB i: {CHROMA_DIR}")
client = Client(host=OLLAMA_HOST)
chroma = PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

# ==========================
# CHUNKING
# ==========================
def semantic_split(text):
    parts = re.split(r'(§\s*\d+|Kapitel\s+\d+|Artikel\s+\d+)', text)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) < MAX_CHARS:
            current += part
        else:
            if len(current.strip()) >= MIN_CHARS:
                chunks.append(current.strip())
            current = part

    if len(current.strip()) >= MIN_CHARS:
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

    final = []
    for ch in chunks:
        if len(ch) <= MAX_CHARS:
            final.append(ch)
        else:
            i = 0
            while i < len(ch):
                part = ch[i:i+MAX_CHARS]
                if len(part) >= MIN_CHARS:
                    final.append(part)
                i += MAX_CHARS - OVERLAP

    return apply_overlap(final)

# ==========================
# TITLE
# ==========================
def extract_title(text):
    for line in text.split("\n"):
        if len(line.strip()) > 10:
            return line[:120]
    return text[:120]

# ==========================
# JSON GROUPING
# ==========================
def extract_from_json(file_path: Path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    def flatten(d, prefix=""):
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    items.update(flatten(v, key))
                else:
                    items[key] = str(v)
        elif isinstance(d, list):
            for i, v in enumerate(d):
                items.update(flatten(v, f"{prefix}[{i}]"))
        return items

    flat = flatten(data)

    chunks = []
    current = ""

    for k, v in flat.items():
        line = f"{k}: {v}"
        if len(v) < 20:
            continue

        if len(current) + len(line) < MAX_CHARS:
            current += line + "\n"
        else:
            if len(current) >= MIN_CHARS:
                chunks.append(current)
            current = line + "\n"

    if len(current) >= MIN_CHARS:
        chunks.append(current)

    return [(c, {"type": "json_group"}) for c in chunks]

# ==========================
# TABLE GROUPING
# ==========================
def table_to_chunks(df, rows_per_chunk=10):
    chunks = []
    cols = list(df.columns)

    for i in range(0, len(df), rows_per_chunk):
        sub = df.iloc[i:i+rows_per_chunk]
        lines = [
            ", ".join(f"{col}: {r[col]}" for col in cols)
            for _, r in sub.iterrows()
        ]
        chunk = "\n".join(lines)
        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)

    return chunks

# ==========================
# EXTRACTORS
# ==========================
def extract_from_pdf(file_path: Path):
    reader = PdfReader(str(file_path))
    items = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) >= MIN_CHARS:
            items.append((text, {"type": "pdf_text", "page": i}))

    if USE_TABULA:
        try:
            tables = tabula.read_pdf(str(file_path), pages='all', multiple_tables=True)
            for ti, table in enumerate(tables):
                if table is not None and not table.empty:
                    for chunk in table_to_chunks(table):
                        items.append((chunk, {"type": "pdf_table", "table_index": ti}))
        except Exception as e:
            print(f"Tabula-fel: {e}")

    return items


def extract_from_csv(file_path: Path):
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)

        chunks = []
        current = ""

        for row in reader:
            line = ", ".join(f"{k}: {v}" for k, v in row.items())

            if len(current) + len(line) < MAX_CHARS:
                current += line + "\n"
            else:
                if len(current) >= MIN_CHARS:
                    chunks.append(current)
                current = line + "\n"

        if len(current) >= MIN_CHARS:
            chunks.append(current)

        return [(c, {"type": "csv_group"}) for c in chunks]


def extract_text(file: Path):
    if file.suffix == ".pdf":
        return extract_from_pdf(file)
    elif file.suffix == ".csv":
        return extract_from_csv(file)
    elif file.suffix == ".json":
        return extract_from_json(file)
    return []

# ==========================
# EMBEDDING
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
# ENRICH
# ==========================
def enrich(text, meta, file):
    title = extract_title(text)

    embedding_text = f"{title}\n{text}"

    display_text = f"""
Titel: {title}
Källa: {file.name}
Typ: {meta.get("type")}
Sida: {meta.get("page", "N/A")}

{text}
"""

    embedding_text = "search_document:\n" + embedding_text

    return embedding_text, display_text

# ==========================
# MAIN
# ==========================
def main():
    docs, metas, ids = [], [], []
    seen = set()

    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue

        print(f"Processar {file.name}")
        items = extract_text(file)

        for text, meta in items:
            for i, ch in enumerate(chunk_text(text)):
                ch = ch.strip()
                if not ch:
                    continue

                emb_text, disp_text = enrich(ch, meta, file)

                if emb_text in seen:
                    continue
                seen.add(emb_text)

                docs.append(disp_text)
                metas.append({
                    "file": file.name,
                    "type": meta.get("type"),
                    "page": meta.get("page", -1),
                    "chunk": i
                })
                ids.append(hashlib.sha1(emb_text.encode()).hexdigest())

    print(f"Chunks: {len(docs)}")

    embeddings = embed_texts([d for d in docs])

    for i in range(0, len(docs), 500):
        collection.add(
            documents=docs[i:i+500],
            embeddings=embeddings[i:i+500],
            metadatas=metas[i:i+500],
            ids=ids[i:i+500]
        )

    print(f"KLAR: {collection.count()} dokument")


if __name__ == "__main__":
    main()
