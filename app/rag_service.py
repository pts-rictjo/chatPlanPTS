from typing import List, Dict, Generator
import os
import re
import time

from flashrank import Ranker, RerankRequest


class RAGService:
    """RAG pipeline: hybrid retrieval -> local FlashRank -> context expansion -> LLM."""

    def __init__(self, rag_system, retriever, bViking=True, bSuppressThink=False):
        self.rag = rag_system
        self.retriever = retriever
        self.bViking = bViking
        self.bSuppressThink = bSuppressThink

        self.top_k = int(os.getenv("UI_TOP_K", "7"))
        self.rerank_k = int(os.getenv("UI_RERANK_K", "30"))
        self.expand_neighbors = int(os.getenv("UI_CONTEXT_NEIGHBORS", "1"))
        self.max_context_chars = int(os.getenv("UI_MAX_CONTEXT_CHARS", "14000"))

        # FlashRank is local and was already part of the previous application.
        # Keep it as the only reranker; no external API is required.
        self.use_reranker = os.getenv("UI_USE_RANKER_MODEL", "true").lower() == "true"
        self.reranker_model = os.getenv(
            "UI_RANKER_MODEL", "ms-marco-MiniLM-L-12-v2"
        )
        self.ranker = None
        if self.use_reranker:
            try:
                self.ranker = Ranker(model_name=self.reranker_model)
                print(f"FlashRank reranker enabled: {self.reranker_model}")
            except Exception as e:
                print(f"FlashRank kunde inte initieras: {e}. Använder RRF utan reranking.")
                self.ranker = None

    def build_retrieval_query(self, query: str, history: List[Dict]) -> str:
        """Add the last user turn only for clearly context-dependent follow-ups.

        This avoids an LLM query-rewrite step while fixing common questions such
        as 'Vilka av dessa är tillståndsfria?'.
        """
        if not history:
            return query

        followup_markers = (
            "dessa", "denna", "detta", "ovan", "föregående", "samma",
            "vilka av", "vad gäller", "hur är det med", "och för", "även",
        )
        q = query.lower().strip()
        if len(q.split()) <= 12 or any(marker in q for marker in followup_markers):
            previous = next(
                (m["content"] for m in reversed(history) if m.get("role") == "user"),
                None,
            )
            if previous and previous.strip() != query.strip():
                return f"{previous}\n\nUppföljande fråga: {query}"
        return query

    def retrieve(self, query: str) -> List[Dict]:
        # Deliberately no LLM-generated query expansion here. It can add noise.
        return self.retriever.hybrid_retrieve(query)

    def dedup_passages(self, passages: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for p in passages:
            doc_id = p.get("id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique.append(p)
        return unique

    @staticmethod
    def _rerank_document(p: Dict) -> str:
        """Create a useful local reranking representation without changing source text."""
        meta = p.get("meta", {})
        labels = []
        for key in ("source", "heading", "page", "type"):
            value = meta.get(key)
            if value not in (None, ""):
                labels.append(f"{key}: {value}")
        prefix = "\n".join(labels)
        text = p.get("text", "")
        return f"{prefix}\n\n{text}" if prefix else text

    def rerank(self, query: str, passages: List[Dict], top_k: int) -> List[Dict]:
        """Rerank the RRF candidates locally using FlashRank."""
        if not passages:
            return []
        if not self.ranker:
            return passages[:top_k]

        candidates = [
            {
                "id": p["id"],
                "text": self._rerank_document(p),
            }
            for p in passages[: self.rerank_k]
        ]

        request = RerankRequest(query=query, passages=candidates)
        ranked = self.ranker.rerank(request)

        by_id = {p["id"]: p for p in passages}
        result = []
        for item in ranked[:top_k]:
            original = by_id.get(item.get("id"))
            if original is None:
                continue
            p = original.copy()
            if "score" in item:
                p["rerank_score"] = float(item["score"])
            result.append(p)
        return result

    def expand_context(self, ranked_passages: List[Dict]) -> List[Dict]:
        """Add neighbouring chunks from the same parent/page.

        The BM25 index already contains the complete chunk corpus, so this does
        not require another vector-database query.
        """
        if not self.expand_neighbors:
            return ranked_passages

        all_docs = getattr(self.rag, "id_to_doc", {})
        by_parent = {}
        for doc_id, doc in all_docs.items():
            meta = doc.get("meta", {})
            parent = meta.get("parent_id")
            if parent:
                by_parent.setdefault(parent, []).append((doc_id, doc))

        for parent, docs in by_parent.items():
            docs.sort(key=lambda x: x[1].get("meta", {}).get("chunk_index", 0))

        result = []
        seen = set()
        for p in ranked_passages:
            pid = p.get("id")
            if pid not in seen:
                result.append(p)
                seen.add(pid)

            meta = p.get("meta", {})
            parent = meta.get("parent_id")
            if not parent or parent not in by_parent:
                continue

            siblings = by_parent[parent]
            pos = next(
                (i for i, (doc_id, _) in enumerate(siblings) if doc_id == pid),
                None,
            )
            if pos is None:
                continue

            for offset in range(1, self.expand_neighbors + 1):
                for idx in (pos - offset, pos + offset):
                    if 0 <= idx < len(siblings):
                        sid, sibling = siblings[idx]
                        if sid in seen:
                            continue
                        result.append({
                            "id": sid,
                            "text": sibling["text"],
                            "meta": sibling["meta"],
                            "source": "context_neighbor",
                        })
                        seen.add(sid)
        return result

    def build_context(self, passages: List[Dict]) -> str:
        parts = []
        total = 0
        source_no = 0
        for p in passages:
            text = p.get("text", "").strip()
            if not text:
                continue
            meta = p.get("meta", {})
            source_no += 1
            source = meta.get("source", "okänd källa")
            page = meta.get("page")
            heading = meta.get("heading")
            location = []
            if page:
                location.append(f"sida {page}")
            if heading:
                location.append(f"avsnitt {heading}")
            where = ", ".join(location)
            label = f"[KÄLLA {source_no}] {source}"
            if where:
                label += f" ({where})"
            block = f"{label}\n{text}"
            if total + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts)

    def get_context(self, query: str, history: List[Dict] | None = None):
        history = history or []
        t0 = time.perf_counter()
        retrieval_query = self.build_retrieval_query(query, history)
        passages = self.retrieve(retrieval_query)
        passages = self.dedup_passages(passages)
        t1 = time.perf_counter()

        best = self.rerank(retrieval_query, passages, self.top_k)
        t2 = time.perf_counter()
        best = self.expand_context(best)
        context = self.build_context(best)
        t3 = time.perf_counter()

        if getattr(self.rag, "debug", False):
            print(
                f"[rag] retrieval={(t1-t0)*1000:.0f} ms, "
                f"rerank={(t2-t1)*1000:.0f} ms, "
                f"context={(t3-t2)*1000:.0f} ms, "
                f"candidates={len(passages)}, final={len(best)}, "
                f"context_chars={len(context)}"
            )
        return context, best

    def stream_answer(
        self, query: str, context: str, history: List[Dict]
    ) -> Generator[str, None, None]:
        system = (
            "Du är en spektrumexpert-assistent. Svara på svenska och använd endast "
            "informationen i KONTEXTEN som sakligt underlag. Om KONTEXTEN inte räcker "
            "för att besvara frågan, säg tydligt att underlaget inte räcker. Hitta inte "
            "på regler, frekvenser, dokument eller slutsatser. Skilj tydligt mellan vad "
            "källan säger och eventuella försiktiga slutsatser. Hänvisa gärna till "
            "[KÄLLA n] när det är relevant."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])
        messages.append({
            "role": "user",
            "content": f"KONTEXT:\n\n{context}\n\nFRÅGA:\n{query}\n\nSVAR:",
        })

        stream = self.rag.ollama.chat(
            model=self.rag.llm_model,
            messages=messages,
            stream=True,
        )
        if self.bSuppressThink:
            response = ""
            for chunk in stream:
                response += chunk["message"]["content"]
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            yield response
        else:
            for chunk in stream:
                yield chunk["message"]["content"]
