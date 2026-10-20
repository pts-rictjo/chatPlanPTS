from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from helper.build_chroma_class import RAGSystem
from .retriever import Retriever
from .rag_service import RAGService
from .conversation_store import ConversationStore


llm_model = os.getenv(
    "UI_LLM_MODEL",
    "vanilj/llama-3.1-instruct-bellman-8b-swedish:q3_k_m",
)

rag_system = RAGSystem(
    data_root="./data",
    chroma_dir="./chroma_db",
    llm_model=llm_model,
)
rag_system.debug = os.getenv("UI_DEBUG_RAGSERVICE", "false").lower() == "true"

retriever = Retriever(
    rag_system,
    dense_k=int(os.getenv("UI_DENSE_K", "30")),
    bm25_k=int(os.getenv("UI_BM25_K", "30")),
    rrf_k=int(os.getenv("UI_RRF_K", "30")),
)
rag_service = RAGService(rag_system, retriever)
conversation_store = ConversationStore()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")


class AskRequest(BaseModel):
    query: str
    conversation_id: str


@app.post("/conversation/new")
async def new_conversation():
    cid = conversation_store.create()
    return {"conversation_id": cid}


@app.post("/ask")
async def ask(req: AskRequest):
    history = conversation_store.get(req.conversation_id)
    context, _best = rag_service.get_context(req.query, history)

    async def generate():
        full_answer = ""
        for token in rag_service.stream_answer(req.query, context, history):
            full_answer += token
            yield token
        conversation_store.append(req.conversation_id, "user", req.query)
        conversation_store.append(req.conversation_id, "assistant", full_answer)

    return StreamingResponse(generate(), media_type="text/plain")
