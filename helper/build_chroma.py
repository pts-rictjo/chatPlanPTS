#!/usr/bin/env python3
"""
Optimerad build_chroma för juridiska dokument, tabeller, CSV och JSON.
Bevarar ALLA rader i CSV-filer och alla textvärden i JSON-filer (≥10 tecken).
Använder semantisk chunking, korrekt Ollama-embeddings, och rik metadata.
"""

import os
import json
import csv
import re
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from chromadb import PersistentClient
from ollama import Client
from pypdf import PdfReader

try:
    import tabula
    USE_TABULA = True
except ImportError:
    USE_TABULA = False
    print("INFO: tabula-py inte installerad. Tabeller från PDF extraheras inte specifikt.")

# ==========================
# KONFIGURATION
# ==========================
DATA_ROOT       = Path(os.getenv("DATA_ROOT", "./data"))
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "spectrum_data")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")          # bäst för svenska

MAX_CHARS   = 800       # max tecken per chunk
MIN_CHARS   = 10        # lägre gräns – behåller nästan allt, filtrerar bara extremt korta/tomma rader
MIN_JSON    = 1         # ta med allt
OVERLAP     = 100       # överlapp mellan delar av samma långa stycke
BATCH_SIZE  = 32        # ChromaDB batch-insättning
MAX_WORKERS = 4         # antal parallella embedding-trådar (1 = sekventiell)

# ==========================
# INIT
# ==========================
print(f"Använder ChromaDB i: {CHROMA_DIR}")
client      = Client(host=OLLAMA_HOST)
chroma      = PersistentClient(path=CHROMA_DIR)
collection  = chroma.get_or_create_collection(name=COLLECTION_NAME)

# ==========================
# SEMANTISK CHUNKING (juridiska markörer)
# ==========================
def semantic_split(text: str) -> list:
    """
    Delar text vid juridiska markörer: §, Kapitel, Artikel.
    Returnerar lista av textsegment (kan vara långa).
    """
    pattern = r'(§\s*\d+|Kapitel\s+\d+|Artikel\s+\d+)'
    parts = re.split(pattern, text)
    chunks = []
    current = ""
    for i in range(len(parts)):
        if re.match(pattern, parts[i]):
            current += parts[i]
        else:
            current += parts[i]
            if len(current) >= MAX_CHARS:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())
    if not chunks:
        return [text.strip()]
    return chunks

def chunk_text(text: str) -> list:
    """
    Dela text med semantisk split, och hantera långa segment med overlap.
    Respekterar MIN_CHARS för att undvika tomma eller extremt korta chunks.
    """
    segments = semantic_split(text)
    final_chunks = []
    for seg in segments:
        if not seg.strip():
            continue
        if len(seg) <= MAX_CHARS:
            if len(seg.strip()) >= MIN_CHARS:
                final_chunks.append(seg.strip())
        else:
            i = 0
            while i < len(seg):
                part = seg[i:i+MAX_CHARS]
                if len(part.strip()) >= MIN_CHARS:
                    final_chunks.append(part.strip())
                i += MAX_CHARS - OVERLAP
    return final_chunks

# ==========================
# EXTRAHERING PER FILTYP
# ==========================
def extract_from_pdf(file_path: Path):
    """Extrahera text och tabeller från PDF. Returnerar lista av (text, metadata)."""
    reader = PdfReader(str(file_path))
    items = []

    # 1. Vanlig text per sida (filtrera bara riktigt korta sidor)
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) >= MIN_CHARS:
            items.append((page_text.strip(), {"type": "pdf_text", "page": page_num}))

    # 2. Tabeller med tabula (om tillgängligt)
    if USE_TABULA:
        try:
            tables = tabula.read_pdf(str(file_path), pages='all', multiple_tables=True)
            for ti, table in enumerate(tables):
                if table is not None and not table.empty:
                    rows_text = []
                    for _, row in table.iterrows():
                        row_str = ", ".join(f"{col}: {row[col]}" for col in table.columns)
                        rows_text.append(row_str)
                    table_text = "\n".join(rows_text)
                    items.append((table_text, {"type": "pdf_table", "table_index": ti}))
        except Exception as e:
            print(f"Varning: tabula misslyckades för {file_path.name}: {e}")

    return items

def extract_from_csv(file_path: Path):
    """
    CSV: spara VARJE rad utan längdfilter.
    Även korta rader (t.ex. en siffra) kan vara meningsfulla.
    """
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        items = []
        for row_num, row in enumerate(reader, start=1):
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
            # Alla rader sparas – inget filter
            items.append((row_text, {"type": "csv_row", "row": row_num}))
        return items

def extract_from_json(file_path: Path):
    """
    JSON: platta ut och skapa en chunk per textvärde.
    Sparar även korta strängar (>=10 tecken) – justera vid behov.
    """
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    def flatten(d, prefix=""):
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    items.update(flatten(v, key))
                elif isinstance(v, str) and len(v) >= MIN_JSON:
                    items[key] = v
        elif isinstance(d, list):
            for idx, elem in enumerate(d):
                items.update(flatten(elem, f"{prefix}[{idx}]"))
        return items

    flat = flatten(data)
    items = [(text, {"type": "json_value", "key": key}) for key, text in flat.items()]
    return items

def extract_text(file: Path):
    """Dispatcher för olika filtyper."""
    if file.suffix == ".pdf":
        return extract_from_pdf(file)
    elif file.suffix == ".csv":
        return extract_from_csv(file)
    elif file.suffix == ".json":
        return extract_from_json(file)
    else:
        return []

# ==========================
# TITEL-EXTRAHERING (för enrich)
# ==========================
def extract_title(text: str) -> str:
    """Första raden som är tillräckligt lång, eller första 120 tecken."""
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) > 20:
            return stripped[:120]
    return text[:120]

def enrich(text: str, meta: dict, file_path: Path) -> tuple:
    """
    Skapar embedding_text (används för vektor) och display_text (lagras i ChromaDB).
    Returnerar (embedding_text, display_text).
    """
    title = extract_title(text)
    source = file_path.name
    page_info = f" (sida {meta.get('page', 'N/A')})" if 'page' in meta else ""
    type_info = meta.get("type", "okänd")
    if 'row' in meta:
        type_info += f" rad {meta['row']}"
    if 'key' in meta:
        type_info += f" nyckel '{meta['key']}'"

    # Texten som skickas till embedding-modellen – prefix för att förbättra retrieval
    embedding_text = f"search_document: {title}\n{text}"

    # Texten som lagras och visas i UI
    display_text = f"""Titel: {title}
Källa: {source}{page_info}
Typ: {type_info}

{text}"""
    return embedding_text, display_text

# ==========================
# PARALLELL EMBEDDING (korrekt)
# ==========================
def embed_texts(texts):
    """Embed en lista med strängar, en i taget, parallellt med ThreadPool."""
    if MAX_WORKERS <= 1:
        embeddings = []
        for t in texts:
            resp = client.embeddings(model=EMBED_MODEL, prompt=t)
            embeddings.append(resp["embedding"])
        return embeddings

    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(client.embeddings, model=EMBED_MODEL, prompt=t): idx
            for idx, t in enumerate(texts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                resp = future.result()
                embeddings[idx] = resp["embedding"]
            except Exception as e:
                print(f"Fel vid embedding av chunk {idx}: {e}")
                # Fallback – nollvektor (bör ej ske)
                embeddings[idx] = [0.0] * 1024   # bge-m3 dimension 1024
    return embeddings

# ==========================
# HUVUDINDEXERING
# ==========================
def main():
    if not DATA_ROOT.exists():
        print(f"FEL: DATA_ROOT = '{DATA_ROOT}' finns inte. Sätt miljövariabeln DATA_ROOT.")
        return

    all_docs = []      # display_text (lagras i ChromaDB)
    all_emb_texts = [] # embedding_text (används för vektor, lagras inte)
    all_metas = []
    all_ids = []

    # Gå igenom alla filer
    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue
        print(f"Processar {file.name}...")
        items = extract_text(file)
        if not items:
            print(f"  Inget extraherbart innehåll i {file.name}")
            continue

        for text_segment, segment_meta in items:
            # Dela segmentet i chunks
            chunks = chunk_text(text_segment)
            for i, chunk_txt in enumerate(chunks):
                if len(chunk_txt) < MIN_CHARS:
                    continue   # extremt kort chunk redan filtrerad, men säkerhetskoll
                # Bygg metadata för denna chunk
                meta = {
                    "file": file.name,
                    "type": segment_meta.get("type", "unknown"),
                    "chunk_index": i,
                }
                if "page" in segment_meta:
                    meta["page"] = segment_meta["page"]
                if "key" in segment_meta:
                    meta["json_key"] = segment_meta["key"]
                if "row" in segment_meta:
                    meta["csv_row"] = segment_meta["row"]
                if "table_index" in segment_meta:
                    meta["table_index"] = segment_meta["table_index"]

                # Skapa embedding_text och display_text
                emb_text, disp_text = enrich(chunk_txt, meta, file)

                # Unikt ID (hash av embedding_text, för att undvika dubletter)
                doc_id = hashlib.sha256(emb_text.encode()).hexdigest()[:32]

                all_emb_texts.append(emb_text)
                all_docs.append(disp_text)
                all_metas.append(meta)
                all_ids.append(doc_id)

    if not all_docs:
        print("Inga dokument hittades. Avslutar.")
        return

    print(f"Skapar embeddings för {len(all_docs)} chunks... (använder {MAX_WORKERS} trådar)")
    embeddings = embed_texts(all_emb_texts)

    # Lägg till i ChromaDB i batcher
    print("Lägger till i ChromaDB...")
    for start in range(0, len(all_docs), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_docs))
        collection.add(
            documents=all_docs[start:end],
            embeddings=embeddings[start:end],
            metadatas=all_metas[start:end],
            ids=all_ids[start:end]
        )
        print(f"  Batch {start//BATCH_SIZE + 1}/{(len(all_docs)-1)//BATCH_SIZE + 1} klar")

    print(f"KLAR! {collection.count()} vektorer i ChromaDB ({CHROMA_DIR})")


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
