from flashrank import Ranker, RerankRequest
from typing import List, Dict, Generator
import uuid
import time
import re
import os

bDebug          = os.getenv("UI_DEBUG_RAGSERVICE", "false").lower() == "true"
ranker_model    = os.getenv("UI_RANKER_MODEL"      , "ms-marco-MiniLM-L-12-v2" )
bUseRanker      = os.getenv("UI_USE_RANKER_MODEL", "false").lower() == "true"

class RAGService :
    def __init__(self, rag_system, retriever , bUseRanker=bUseRanker, bViking=True , bExpanded = False , bSuppressThink=False ):
        self.rag            = rag_system
        self.retriever      = retriever
        self.bUseRanker     = bUseRanker
        self.bViking        = bViking       # Kommer använda nordiska språkstödsmodeller
        self.bExpanded      = bExpanded
        self.bSuppressThink = bSuppressThink # Undertrycker think i output
        if bUseRanker:
            self.ranker = Ranker(model_name=ranker_model)
        else:
            self.ranker = None
        self.n_results  = 50   # recall
        self.top_k      = 7    # efter rerank

    def expand_query_expert(self, query: str) -> List[str]:
        if hasattr(self.rag, "questions_") and self.rag.questions_:
            return [query] + self.rag.questions_
        return [query]

    def expand_query(self, query: str) -> List[str]:
        # For minimal debugging, disable expansion
        return [query]

    def embed_query(self, query: str):
        if self.bExpanded:
            enriched = f"QUERY:\n{query}"
        else:
            enriched = query
        return self.rag.generate_embedding(enriched)

    def retrieve(self, query: str) -> List[Dict]:
        """Använd hybridretriever med expanded queries."""
        if self.bExpanded:
            queries = self.expand_query_expert(query)
        else:
            queries = self.expand_query(query)
        if bDebug :
            print(f"[DEBUG] Expanded to {len(queries)} queries: {queries}")
        all_passages = []
        for q in queries:
            # Hämta med hybrid (dense + BM25)
            passages = self.retriever.hybrid_retrieve(q, top_k=self.n_results)
            all_passages.extend(passages)
        return all_passages

    def dedup_passages(self, passages: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for p in passages:
            h = hash(p["text"])
            if h not in seen:
                seen.add(h)
                unique.append(p)
        return unique

    def rerank(self, query: str, passages: List[Dict], top_k: int = 7) -> List[Dict]:
        # FlashRank förväntar sig fälten "id", "text", "meta"
        request = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(request)
        return ranked[:top_k]

    def build_context_untruncated(self, ranked_passages: List[Dict]) -> str:
        return "\n\n".join([p["text"] for p in ranked_passages])

    def build_context(self, ranked_passages: List[Dict], max_chars: int = 12000) -> str:
        parts = []
        total = 0
        for p in ranked_passages:
            text = p["text"]
            if total + len(text) > max_chars:
                # Only add whole passages, stop before exceeding limit
                if parts:  # already have some context
                    break
                else:      # first passage alone is too long, truncate it with warning
                    parts.append(text[:max_chars] + "...")
                    break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    def get_context(self, query: str) -> tuple[str, List[Dict]]:
        if bDebug :
            print(f"\n[DEBUG] Query: {query}")
            t0 = time.time()
        passages = self.retrieve(query)
        if bDebug :
            print(f"[DEBUG] Retrieved {len(passages)} passages in {time.time()-t0:.2f}s")
            t1 = time.time()
        passages = self.dedup_passages(passages)
        if bDebug :
            print(f"[DEBUG] After dedup: {len(passages)} passages in {time.time()-t1:.2f}s")

        if self.bUseRanker:
            t2 = time.time()
            best = self.rerank(query, passages, top_k=self.top_k)
            if bDebug :
                print(f"[DEBUG] After rerank: {len(best)} passages in {time.time()-t2:.2f}s")
        else:
            best = passages[:self.top_k]
            if bDebug:
                print(f"[DEBUG] Using top {len(best)} hybrid passages (no rerank)")
        #context = self.build_context_untruncated(best)
        context = self.build_context(best, max_chars=12000)
        if bDebug :
            print(f"[DEBUG] Context length: {len(context)} chars")
        return context, best

    def stream_answer(self, query: str, context: str, history: List[Dict]) -> Generator[str, None, None]:
        """Anropa Ollama via RAGSystem.ollama-klienten."""
        first = True
        if bDebug:
            print(f"[DEBUG] Starting Ollama stream...")
            t0 = time.time()
        if self.bViking :
            messages = [
                {"role": "system", "content": "Du är en spektrum expert assistent som svarar med hjälp av den bifogade informationen. Om svaret inte finns i informationen förtydligar du att det inte finns det bifogade dokumenten."}
            ]
            messages.extend(history)
            messages.append({
                "role": "user",
                "content": f"Baserat på följande information:\n\n{context}\n\nFråga: {query}\n\nSvar:"
            })
            if bDebug :
                print('##ANVÄNDER##')
                print(context)
                print('############')
        else :
            messages = [
                {"role": "system", "content": "Answer based on the provided context only."}
            ]
            messages.extend(history)
            messages.append({
                "role": "user",
                "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"
            })
        stream = self.rag.ollama.chat(
            model = self.rag.llm_model,
            messages = messages,
            stream = True
        )

        if self.bSuppressThink :
            # INGEN STREAMING
            response = ""
            for chunk in stream:
                response += chunk["message"]["content"]
            response = re.sub(
                r"<think>.*?</think>",
                "",
                response,
                flags=re.DOTALL
            )
            yield response
        else:
            for chunk in stream:
                if first and bDebug:
                    print(f"[DEBUG] First token after {time.time()-t0:.2f}s")
                    first = False
                yield chunk["message"]["content"]
