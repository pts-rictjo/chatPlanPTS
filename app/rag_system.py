from fastapi import FastAPI
from pydantic import BaseModel
import os
import chromadb
from chromadb.config import Settings
from ollama import Client

# ==========================
# CONFIG
# ==========================

CHROMA_DIR = os.getenv("CHROMA_DIR", "/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "spectrum_data")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

TOP_K = int(os.getenv("TOP_K", "5"))

# ==========================
# INIT
# ==========================
app = FastAPI(title="Spectrum RAG API")
ollama = Client(host=OLLAMA_HOST)
chroma_client = chromadb.Client(
    Settings(persist_directory=CHROMA_DIR)
)

collection = chroma_client.get_collection(name=COLLECTION_NAME)

# ==========================
# SCHEMA
# ==========================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: list

# ==========================
# CORE
# ==========================
def embed(text: str):
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]

def retrieve(query: str):
    emb = embed(query)

    results = collection.query(
        query_embeddings=[emb],
        n_results=TOP_K
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for d, m in zip(docs, metas):
        src = m.get("file", "okänd")
        context_blocks.append(f"[{src}] {d}")

    return context_blocks

def generate_answer(question: str, context_blocks: list[str]):
    context = "\n\n".join(context_blocks)

    prompt = f"""
Du är expert på frekvensspektrum.

Använd ENDAST information från kontexten.
Svara kort och korrekt.

KONTEXT:
{context}

FRÅGA:
{question}

SVAR:
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ==========================
# ENDPOINT
# ==========================

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    context_blocks = retrieve(req.question)
    answer = generate_answer(req.question, context_blocks)

    return {
        "answer": answer,
        "context": context_blocks
    }

@app.get("/health")
def health():
    return {"status": "ok"}
