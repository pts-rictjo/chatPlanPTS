#!/usr/bin/env python3
"""
build_chroma v9 + Whoosh (hybrid RAG) + query expansion
- PMI‑baserad ontologi med bigrams (potentiellt störande)
- Extraherar automatiskt ordpar som "fast_radio", "data_överföring"
- Bygger Whoosh‑index för nyckelordssökning
- Lagrar freq_min / freq_max i båda databaserna
"""

import os, re, csv, json, uuid
import html
import math
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from chromadb import PersistentClient
from ollama import Client
from pypdf import PdfReader

# Whoosh (keyword search)
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.analysis import StemmingAnalyzer

try:
    import tabula
    USE_TABULA = True
except ImportError:
    USE_TABULA = False

# ==========================
# KONFIGURATION
# ==========================
DATA_ROOT           = Path(os.getenv("DATA_ROOT"    , "./data"))
CHROMA_DIR          = os.getenv("CHROMA_DIR"        , "./chroma_db")
WHOOSH_DIR          = os.getenv("WHOOSH_DIR"        , "./chroma_db/whoosh_index")
COLLECTION_NAME     = os.getenv("COLLECTION_NAME"   , "spectrum_data")
OLLAMA_HOST         = os.getenv("OLLAMA_HOST"       , "http://127.0.0.1:11434")
EMBED_MODEL         = os.getenv("EMBED_MODEL"       , "nomic-embed-text-v2-moe")

MAX_CHARS           = 1000
OVERLAP             = 200
MIN_CHARS           = 3
BATCH_SIZE          = 32
MAX_WORKERS         = 4

# Stoppordslista (oförändrad från v9)
STOPWORDS = {
    "och","eller","med","för","som","den","det","att","av","till","i","på","är","en","ett","sig","om","under","efter","över","vid",
}

# ==========================
# RENGÖRING OCH CHUNKING
# ==========================
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(nan|none|null)\b", "", text, flags=re.IGNORECASE)
    for unit in ['kHz','MHz','GHz']:
        text = re.sub(rf'(\d+),(\d+)\s*{unit}', r'\1.\2 ' + unit, text)
        text = re.sub(rf'(\d)\.(\d{{3}})\s*{unit}', r'\1\2 ' + unit, text)
    return text.strip()

def chunk_text(text: str):
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if len(text) < MAX_CHARS and len(text) >= MIN_CHARS:
        return [text]
    out = []
    i = 0
    while i < len(text):
        part = text[i:i+MAX_CHARS]
        if len(part) >= MIN_CHARS:
            out.append(part)
        i += MAX_CHARS - OVERLAP
    return out

# ==========================
# FREKVENSBEHANDLING
# ==========================
def normalize_freq_val(freq_str: str) -> int | None:
    freq_str = re.sub(r'\s+', '', freq_str.lower())
    m = re.match(r'(\d+(?:[.,]\d+)?)(khz|mhz|ghz)', freq_str)
    if not m:
        return None
    val, unit = m.groups()
    val = float(val.replace(',', '.'))
    if unit == 'ghz':
        val *= 1000
    elif unit == 'khz':
        val /= 1000
    return int(round(val))

def extract_ranges(text: str) -> list[tuple[int, int]]:
    pattern = r'(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(MHz|GHz)'
    matches = re.findall(pattern, text, re.I)
    ranges = []
    for a, b, unit in matches:
        a = float(a.replace(',', '.'))
        b = float(b.replace(',', '.'))
        if unit.lower() == 'ghz':
            a *= 1000
            b *= 1000
        ranges.append((int(round(a)), int(round(b))))
    return ranges

def extract_all_frequencies(text: str) -> list[int]:
    point_matches = re.findall(r'\b\d+(?:[.,]\d+)?\s*(?:khz|mhz|ghz)\b', text, re.I)
    values = []
    for p in point_matches:
        v = normalize_freq_val(p)
        if v is not None:
            values.append(v)
    for lo, hi in extract_ranges(text):
        values.append(lo)
        values.append(hi)
    values = [v for v in values if 1 <= v <= 300000]
    seen = set()
    uniq = []
    for v in values:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

def extract_features_with_bigrams(text: str):
    t = text.lower()
    t = re.sub(r'[^a-zåäö0-9\s]', ' ', t)
    freqs = extract_all_frequencies(t)
    words = re.findall(r'\b[a-zåäö]{3,}\b', t)
    forbidden_units = {"khz", "mhz", "ghz", "hz", "db", "kw", "w", "v", "a"}
    forbidden = STOPWORDS.union(forbidden_units, {"none","nan","null"})
    words = [w for w in words if w not in forbidden]
    # bigrams
    bigrams = []
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if len(w1) > 3 and len(w2) > 3:
            bigrams.append(f"{w1}_{w2}")
    all_terms = words + bigrams
    return freqs, all_terms

# ==========================
# FREQUENCY BOOST
# ==========================
def freq_boost(query_freqs, meta):
    if not query_freqs:
        return 0.0

    doc_freqs = meta.get("freqs") or []
    if not doc_freqs:
        return 0.0

    score = 0.0
    for qf in query_freqs:
        for df in doc_freqs:
            dist = abs(qf - df)
            score += math.exp(-dist / 200)  # decay

    return score / (len(doc_freqs) + 1)

# ==========================
# EXTRAHERARE (PDF, CSV, JSON)
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

def extract_from_pdf(file_path):
    items = []
    reader = PdfReader(str(file_path))
    for i, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if len(txt) > MIN_CHARS:
            items.append((txt, {"type": "pdf_text", "page": i}))
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
            print(f"⚠️ Tabula error: {e}")
    return items

def extract_from_csv(file_path):
    items = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            fields = []
            for k, v in row.items():
                v = clean_text(str(v))
                if v and v.lower() not in ["nan", "none"]:
                    fields.append(f"{k}: {v}")
            if fields:
                txt = "Frekvensdata | " + " | ".join(fields)
                items.append((txt, {"type": "csv_row", "row": i}))
    return items

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
# WHOOSH (INIT, INDEX)
# ==========================
def init_whoosh():
    """Skapar eller öppnar Whoosh‑indexet."""
    if not os.path.exists(WHOOSH_DIR):
        os.mkdir(WHOOSH_DIR)
    schema = Schema(
        id=ID(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer()),
        freq_min=NUMERIC(stored=True),
        freq_max=NUMERIC(stored=True),
    )
    if not index.exists_in(WHOOSH_DIR):
        return index.create_in(WHOOSH_DIR, schema)
    return index.open_dir(WHOOSH_DIR)

# ==========================
# EMBEDDING & ONTOLOGI
# ==========================
def build_embedding_text_w_ontology(text, ontology, meta):
    freqs, _ = extract_features_with_bigrams(text)
    hints = []
    for f in freqs:
        hints.extend(ontology.get(f, []))
    hint_str = " ".join(set(hints))
    if meta.get("type") in ["pdf_table", "csv_row"]:
        hint_str += " strukturerad frekvensdata tabell"
    freq_str = " ".join(str(f) for f in freqs)
    return f"""
search_document:
{text}

frekvenser:
{freq_str}

semantiska_termer:
{hint_str}
"""

def build_embedding_text(text, ontology, meta):
    freqs, _ = extract_features_with_bigrams(text)
    freq_str = " ".join(str(f) for f in freqs)
    if meta.get("type") in ["pdf_table", "csv_row"]:
        text = "Structured data: " + text
    return f"{text} Frequencies: {freq_str}"

def embed_texts(texts, ollama_client):
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(ollama_client.embeddings, model=EMBED_MODEL, prompt=t): i
            for i, t in enumerate(texts)
        }
        for f in as_completed(futures):
            i = futures[f]
            try:
                results[i] = f.result()["embedding"]
            except Exception as e:
                print(f"❌ Embed error {i}: {e}")
                results[i] = [0.0] * 768
    return results

def build_inverted_ontology(ontology, chroma_dir):
    inverted = defaultdict(list)
    for freq, terms in ontology.items():
        for term in terms:
            inverted[term].append(freq)
    for term in inverted:
        inverted[term].sort()
    inverted_display = {k.replace("_", " "): v for k, v in inverted.items()}
    with open(f"{chroma_dir}/inverted_ontology.json", "w", encoding="utf-8") as f:
        json.dump(inverted_display, f, indent=2, ensure_ascii=False)
    print(f"✅ Inverterad ontologi sparad: {len(inverted)} termer")

# ==========================
# HUVUDFUNKTION: create_db()
# ==========================
def create_db():
    print("🔌 Initierar ...")
    ollama = Client(host=OLLAMA_HOST)
    chroma = PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    # Initiera Whoosh
    ix = init_whoosh()
    writer = ix.writer()

    print("🧠 Läser dokument och bygger statistik ...")
    all_chunks = []
    co_freq = defaultdict(Counter)
    freq_chunk_count = Counter()
    global_term_count = Counter()

    for file in DATA_ROOT.rglob("*"):
        if not file.is_file():
            continue
        print(f"📄 {file.name}")
        items = extract(file)
        for text, meta in items:
            chunks = chunk_text(text)
            for ch in chunks:
                if not ch.strip():
                    continue
                freqs, terms = extract_features_with_bigrams(ch)
                all_chunks.append((ch, meta, file.name))
                if not freqs or not terms:
                    continue
                unique_freqs = set(freqs)
                for f in unique_freqs:
                    freq_chunk_count[f] += 1
                for t in set(terms):
                    global_term_count[t] += 1
                for f in unique_freqs:
                    for t in terms:
                        co_freq[f][t] += 1

    if not all_chunks:
        print("❌ Inga chunks hittades.")
        return

    total_chunks = len(all_chunks)
    print(f"📊 Totalt {total_chunks} chunks.")
    print(f"   Frekvenser funna: {len(freq_chunk_count)}")
    print(f"   Termer (ord+bigram) funna: {len(global_term_count)}")

    # Bygg ontologi med PMI
    print("🔗 Beräknar PMI och skapar ontologi ...")
    ontology = {}
    for freq, term_counter in co_freq.items():
        f_total = freq_chunk_count.get(freq, 1)
        scored = []
        for term, co_occ in term_counter.items():
            t_total = global_term_count.get(term, 1)
            if t_total == 0 or f_total == 0:
                continue
            pmi = math.log((co_occ / f_total) / (t_total / total_chunks) + 1e-12)
            min_co_occ  = 1 if "_" in term else 2
            min_pm      = 0.2 if "_" in term else 0.5
            if co_occ >= min_co_occ and pmi > min_pm:
                scored.append((term, pmi))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _ in scored[:10]]
        if top_terms:
            ontology[freq] = top_terms

    build_inverted_ontology(ontology, CHROMA_DIR)
    print(f"✅ Ontologi innehåller {len(ontology)} frekvenser.")
    sample = list(ontology.items())[:5]
    for freq, terms in sample:
        display_terms = [t.replace("_", " ") for t in terms[:5]]
        print(f"   {freq} MHz: {', '.join(display_terms)}")

    # Bygg vektordatabas OCH Whoosh
    print("📦 Bygger ChromaDB och Whoosh ...")
    docs, emb_texts, metas, ids = [], [], [], []
    seen = set()

    for text, meta, fname in all_chunks:
        key = key = hash(text) # text.lower()
        if key in seen:
            continue
        seen.add(key)

        # Beräkna frekvenser för denna chunk
        freqs = extract_all_frequencies(text)
        freq_min = min(freqs) if freqs else 0
        freq_max = max(freqs) if freqs else 0

        # Chroma‑dokument
        doc_text = f"Källa: {fname}\n{text}"
        docs.append(doc_text)
        emb_texts.append(build_embedding_text(text, ontology, meta))

        # Metadata (lägg till freq_min, freq_max)
        meta_with_freq = {
            "file": fname,
            "freqs": ",".join(map(str, freqs)) ,
            "freq_min": freq_min,
            "freq_max": freq_max,
            **meta
        }
        metas.append(meta_with_freq)
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)

        # Whoosh – lägg till dokument
        writer.add_document(
            id=doc_id,
            content=text + " " + " ".join(map(str, freqs)) ,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    # Spara Whoosh‑indexet
    writer.commit()
    print(f"✅ Whoosh-index sparat i {WHOOSH_DIR}")

    # Skapa embeddings för Chroma
    print(f"🧠 Skapar embeddings för {len(docs)} chunks ...")
    vectors = embed_texts(emb_texts, ollama)

    print("💾 Skriver till ChromaDB ...")
    for i in range(0, len(docs), BATCH_SIZE):
        collection.add(
            documents=docs[i:i+BATCH_SIZE],
            embeddings=vectors[i:i+BATCH_SIZE],
            metadatas=metas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
        print(f"   Batch {i//BATCH_SIZE + 1} / {(len(docs)-1)//BATCH_SIZE + 1}")

    # Spara konfiguration
    with open(f"{CHROMA_DIR}/chromadb_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "COLLECTION_NAME": COLLECTION_NAME,
            "EMBED_MODEL": EMBED_MODEL,
            "MAX_CHARS": MAX_CHARS,
            "OVERLAP": OVERLAP,
            "MIN_CHARS": MIN_CHARS,
            "BATCH_SIZE": BATCH_SIZE,
            "WHOOSH_DIR":WHOOSH_DIR,
        }, f, indent=2)

    print(f"\n✅ KLART! {collection.count()} vektorer i {CHROMA_DIR}")
    print(f"   Whoosh-index i {WHOOSH_DIR}")

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    create_db()
