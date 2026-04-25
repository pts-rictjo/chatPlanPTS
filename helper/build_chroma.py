#!/usr/bin/env python3
"""
build_chroma v5 – större chunks, meningbaserad delning
"""

import os
import json
import csv
import re
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from chromadb import PersistentClient
from ollama import Client
from pypdf import PdfReader

try:
    import tabula
    USE_TABULA = True
except ImportError:
    USE_TABULA = False

# ==========================
# CONFIG
# ==========================
DATA_ROOT           = Path(os.getenv("DATA_ROOT", "./data"))
CHROMA_DIR          = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME     = os.getenv("COLLECTION_NAME", "spectrum_data")
OLLAMA_HOST         = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL         = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")

MAX_CHARS       = 1000
OVERLAP         = 200
MIN_CHARS       = 3         # lägsta längd för att spara en chunk
BATCH_SIZE      = 32
MAX_WORKERS     = 4

# ==========================
# INIT
# ==========================
print(f"ChromaDB path: {CHROMA_DIR}")
client = Client(host=OLLAMA_HOST)
chroma = PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

# ==========================
# CLEANING
# ==========================
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(nan|none|null)\b", "", text, flags=re.IGNORECASE)
    return text.strip()

# ==========================
# FÖRBÄTTRAD CHUNKING (meningbaserad)
# ==========================
def chunk_text(text: str):
    text = clean_text(text)
    if not text:
        return []

    # Försök dela på meningar
    sentences = re.split(r'(?<=[.!?:;])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= MAX_CHARS:
            current += sent + " "
        else:
            if current:
                chunks.append(current.strip())
            # Om enskild mening är för lång, dela med overlap
            if len(sent) > MAX_CHARS:
                i = 0
                while i < len(sent):
                    part = sent[i:i+MAX_CHARS]
                    if len(part) >= MIN_CHARS:
                        chunks.append(part.strip())
                    i += MAX_CHARS - OVERLAP
                current = ""
            else:
                current = sent + " "

    if current:
        chunks.append(current.strip())

    # Filtrera bort för korta chunks
    return [c for c in chunks if len(c) >= MIN_CHARS]

# ==========================
# TABLE FORMAT (oförändrad, bra)
# ==========================
def format_table_row(row, columns, source):
    fields = []
    for col in columns:
        val = str(row[col]).strip()
        if val and val.lower() not in ["nan", "none", ""]:
            fields.append(f"{col}: {val}")
    if not fields:
        return None
    return f"Frekvensdata | Källa: {source} | " + " | ".join(fields)

# ==========================
# PDF-extrahering (oförändrad)
# ==========================
def extract_from_pdf(file_path):
    items = []
    reader = PdfReader(str(file_path))

    # Text per sida
    for i, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if len(txt) > MIN_CHARS:
            items.append((txt, {"type": "pdf_text", "page": i}))

    # Tabeller
    if USE_TABULA:
        try:
            tables = tabula.read_pdf(str(file_path), pages="all", multiple_tables=True)
            for ti, table in enumerate(tables):
                if table is None or table.empty:
                    continue
                for _, row in table.iterrows():
                    row_txt = format_table_row(row, table.columns, file_path.name)
                    if row_txt:
                        items.append((row_txt, {"type": "pdf_table", "table": ti}))
        except Exception as e:
            print(f"❌ Tabula error: {e}")
    return items

# ==========================
# CSV-extrahering
# ==========================
def extract_from_csv(file_path):
    items = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            fields = []
            for k, v in row.items():
                v = str(v).strip()
                if v and v.lower() not in ["nan", "none"]:
                    fields.append(f"{k}: {v}")
            if fields:
                txt = "Frekvensdata | " + " | ".join(fields)
                items.append((txt, {"type": "csv_row", "row": i}))
    return items

# ==========================
# JSON-extrahering
# ==========================
def extract_from_json(file_path):
    items = []
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            txt = clean_text(obj)
            if len(txt) > MIN_CHARS:
                items.append((txt, {"type": "json", "key": prefix}))
    walk(data)
    return items

def extract(file):
    if file.suffix == ".pdf":
        return extract_from_pdf(file)
    if file.suffix == ".csv":
        return extract_from_csv(file)
    if file.suffix == ".json":
        return extract_from_json(file)
    return []

# ==========================
# QUERY-AWARE EMBEDDING TEXT ; ADD DICTIONARY HINTS HERE LATER
# ==========================
def build_embedding_text(text, meta):
    hints = []
    t = text.lower()
    if meta.get("type") in ["pdf_table", "csv_row"]:
        hints.append("frekvensband spektrum radio användning")
    if "rlan" in t or "was" in t:
        hints.append("wifi WAS RLAN trådlöst nätverk")
    if "khz" in t or "mhz" in t or "ghz" in t or "thz" in t:
        hints.append("radiofrekvens band spektrum")
    if "tillstånd" in t or "not 1" in t:
        hints.append("regler licens undantag tillståndsplikt")
    hint_str = " ".join(hints)
    return f"search_document: {hint_str} {text}"

# ==========================
# EMBEDDING
# ==========================
def embed(texts):
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(client.embeddings, model=EMBED_MODEL, prompt=t): i for i, t in enumerate(texts)}
        for f in as_completed(futures):
            i = futures[f]
            try:
                results[i] = f.result()["embedding"]
            except Exception as e:
                print(f"❌ Embed error {i}: {e}")
                results[i] = [0.0] * 768
    return results

# ==========================
# MAIN
# ==========================
def main():
    docs, emb_texts, metas, ids = [], [], [], []
    seen = set()
    total_raw = 0

    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue
        print(f"📄 {file.name}")
        items = extract(file)
        for text, meta in items:
            total_raw += 1
            if meta.get("type") in ["pdf_table", "csv_row"]:
                chunks = [text]
            else:
                chunks = chunk_text(text)
            for ch in chunks:
                if not ch:
                    continue
                key = ch.lower()
                if key in seen:
                    continue
                seen.add(key)
                emb_text = build_embedding_text(ch, meta)
                doc_text = f"Källa: {file.name}\nTyp: {meta.get('type')}\n\n{ch}".strip()
                docs.append(doc_text)
                emb_texts.append(emb_text)
                metas.append({"file": file.name, **meta})
                ids.append(str(uuid.uuid4()))

    print(f"\nRaw segments: {total_raw} | Unique chunks: {len(docs)}")
    if not docs:
        print("❌ No data")
        return

    print(f"\nEmbedding {len(docs)} chunks...")
    vectors = embed(emb_texts)

    print("Writing to Chroma...")
    for i in range(0, len(docs), BATCH_SIZE):
        collection.add(
            documents=docs[i:i+BATCH_SIZE],
            embeddings=vectors[i:i+BATCH_SIZE],
            metadatas=metas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
        print(f"Batch {i//BATCH_SIZE + 1} done")

    print(f"\n✅ DONE: {collection.count()} chunks stored\n{CHROMA_DIR}")

    ofile = open( f"{CHROMA_DIR}/chromadb_params.json","w")
    print( "[{" + f"COLLECTION_NAME:\"{COLLECTION_NAME}\"\nEMBED_MODEL:\"{EMBED_MODEL}\"\nMAX_CHARS:{MAX_CHARS}\nOVERLAP:{OVERLAP}\nMIN_CHARS:{MIN_CHARS}\nBATCH_SIZE:{BATCH_SIZE}\n"+"}]" , file=ofile )
    ofile.close()

if __name__ == "__main__":
    main()
