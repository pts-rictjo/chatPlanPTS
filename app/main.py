from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import sys
from pathlib import Path

# Lägg till så att vi kan importera helper
sys.path.append(str(Path(__file__).parent.parent))

from helper.build_chroma_class  import RAGSystem
from .retriever                 import Retriever
from .rag_service               import RAGService
from .conversation_store        import ConversationStore

import os

bDebug      = os.getenv("UI_DEBUG_RAGSERVICE", "false").lower() == "true"
llm_model   = os.getenv("UI_LLM_MODEL"       , "vanilj/llama-3.1-instruct-bellman-8b-swedish:q3_k_m" ) # "qwen3:8b" "qwen3:4b" "akx/viking-7b:latest"

# ---------- Initiera lager med RAGSystem ----------
rag_system = RAGSystem(
    data_root  = "./data",
    chroma_dir = "./chroma_db",
    llm_model  = llm_model,
)
# Se till att BM25 laddas (klass gör det i __init__ via load_bm25())
# Om inte, anropa rag_system.load_bm25() explicit.

retriever = Retriever(rag_system)
rag_service = RAGService(rag_system, retriever)
conversation_store = ConversationStore()

# ---------- FastAPI ----------
# Här exponerar vi olika publika källor
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

class AskRequest(BaseModel):
    query: str
    conversation_id: str

@app.get("/")
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/conversation/new")
async def new_conversation():
    cid = conversation_store.create()
    return {"conversation_id": cid}

@app.post("/ask")
async def ask(req: AskRequest):
    history = conversation_store.get(req.conversation_id)
    context, best = rag_service.get_context(req.query)
    if bDebug:
        print(f"CONTEXT: {context[:500]}...")
    async def generate():
        full_answer = ""
        for token in rag_service.stream_answer(req.query, context, history):
            full_answer += token
            yield token
        # Spara efter streaming
        conversation_store.append(req.conversation_id, "user", req.query)
        conversation_store.append(req.conversation_id, "assistant", full_answer)

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False )
