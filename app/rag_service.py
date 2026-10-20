from typing import Dict, Generator, List
import os
import re
import time

try:
    from flashrank import Ranker, RerankRequest
except ImportError:  # pragma: no cover - handled at runtime with a clear fallback
    Ranker = None
    RerankRequest = None


class RAGService:
    """RAG pipeline: hybrid retrieval -> local rerank -> context expansion -> LLM."""

    def __init__(self, rag_system, retriever, bViking=True, bSuppressThink=False):
        self.rag = rag_system
        self.retriever = retriever
        self.bViking = bViking
        self.bSuppressThink = bSuppressThink

        self.top_k = int(os.getenv("UI_TOP_K", "7"))
        self.rerank_k = int(os.getenv("UI_RERANK_K", "30"))
        self.expand_neighbors = int(os.getenv("UI_CONTEXT_NEIGHBORS", "1"))
        self.max_context_chars = int(os.getenv("UI_MAX_CONTEXT_CHARS", "14000"))

        self.use_reranker = (
            os.getenv("UI_USE_RANKER_MODEL", "true").lower() == "true"
        )
        self.reranker_model = os.getenv(
            "UI_RANKER_MODEL", "ms-marco-MiniLM-L-12-v2"
        )
        self.ranker = None

        if self.use_reranker:
            if Ranker is None:
                print("FlashRank saknas; använder RRF-resultat utan reranking.")
                self.use_reranker = False
            else:
                try:
                    self.ranker = Ranker(model_name=self.reranker_model)
                    print(f"FlashRank reranker enabled: {self.reranker_model}")
                except Exception as exc:
                    print(
                        f"Kunde inte initiera FlashRank ({exc}); "
                        "använder RRF-resultat utan reranking."
                    )
                    self.ranker = None
                    self.use_reranker = False

    def build_retrieval_query(self, query: str, history: List[Dict]) -> str:
        """Use the previous user turn only for clearly context-dependent follow-ups.

        No LLM-generated query expansion is used here. This keeps retrieval
        deterministic and avoids introducing unrelated terms into BM25/dense
        search.
        """
        if not history:
            return query

        followup_markers = (
            "dessa",
            "denna",
            "detta",
            "ovan",
            "föregående",
            "samma",
            "vilka av",
            "vad gäller",
            "hur är det med",
            "och för",
            "även",
        )
        q = query.lower().strip()
        if len(q.split()) <= 12 or any(marker in q for marker in followup_markers):
            previous = next(
                (
                    message["content"]
                    for message in reversed(history)
                    if message.get("role") == "user"
                    and message.get("content", "").strip()
                ),
                None,
            )
            if previous and previous.strip() != query.strip():
                return f"{previous}\n\nUppföljande fråga: {query}"

        return query

    def retrieve(self, query: str) -> List[Dict]:
        return self.retriever.hybrid_retrieve(query)

    @staticmethod
    def dedup_passages(passages: List[Dict]) -> List[Dict]:
        """Deduplicate by stable chunk ID."""
        seen = set()
        unique = []
        for passage in passages:
            doc_id = passage.get("id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            unique.append(passage)
        return unique

    @staticmethod
    def _rerank_document(passage: Dict) -> str:
        """Give FlashRank source metadata as lightweight retrieval context."""
        meta = passage.get("meta", {})
        labels = []
        for key in ("source", "heading", "page", "section", "type"):
            value = meta.get(key)
            if value not in (None, ""):
                labels.append(f"{key}: {value}")

        prefix = "\n".join(labels)
        text = passage.get("text", "")
        return f"{prefix}\n\n{text}" if prefix else text

    def rerank(self, query: str, passages: List[Dict], top_k: int) -> List[Dict]:
        """Rerank RRF candidates locally with FlashRank.

        FlashRank versions differ slightly in their returned dictionary fields,
        so both returned IDs and returned positions are handled defensively.
        """
        if not passages:
            return []
        if not self.ranker or not self.use_reranker:
            return passages[:top_k]

        candidates = passages[: self.rerank_k]
        rerank_passages = [
            {"id": p["id"], "text": self._rerank_document(p)}
            for p in candidates
        ]

        try:
            request = RerankRequest(query=query, passages=rerank_passages)
            ranked = self.ranker.rerank(request)
        except Exception as exc:
            print(f"FlashRank rerank misslyckades ({exc}); använder RRF.")
            return candidates[:top_k]

        by_id = {p["id"]: p for p in candidates}
        result = []

        for position, item in enumerate(ranked):
            if not isinstance(item, dict):
                continue

            doc_id = item.get("id")
            original = by_id.get(doc_id) if doc_id is not None else None

            # Some FlashRank versions may return a position rather than an ID.
            if original is None:
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(candidates):
                    original = candidates[index]

            if original is None and position < len(candidates):
                # Last-resort compatibility fallback. Normally FlashRank
                # returns the candidate ID.
                original = candidates[position]

            if original is None:
                continue

            enriched = original.copy()
            if item.get("score") is not None:
                enriched["rerank_score"] = float(item["score"])
            elif item.get("relevance_score") is not None:
                enriched["rerank_score"] = float(item["relevance_score"])
            result.append(enriched)

            if len(result) >= top_k:
                break

        return result or candidates[:top_k]

    def expand_context(self, ranked_passages: List[Dict]) -> List[Dict]:
        """Add neighbouring chunks from the same parent without another search."""
        if self.expand_neighbors <= 0:
            return ranked_passages

        all_docs = getattr(self.rag, "id_to_doc", {}) or {}
        by_parent = {}

        for doc_id, doc in all_docs.items():
            meta = doc.get("meta", {})
            parent = meta.get("parent_id")
            if parent:
                by_parent.setdefault(parent, []).append((doc_id, doc))

        for docs in by_parent.values():
            docs.sort(key=lambda item: item[1].get("meta", {}).get("chunk_index", 0))

        result = []
        seen = set()

        for passage in ranked_passages:
            doc_id = passage.get("id")
            if doc_id and doc_id not in seen:
                result.append(passage)
                seen.add(doc_id)

            meta = passage.get("meta", {})
            parent = meta.get("parent_id")
            siblings = by_parent.get(parent)
            if not siblings or not doc_id:
                continue

            position = next(
                (i for i, (sid, _) in enumerate(siblings) if sid == doc_id),
                None,
            )
            if position is None:
                continue

            for offset in range(1, self.expand_neighbors + 1):
                for index in (position - offset, position + offset):
                    if not 0 <= index < len(siblings):
                        continue
                    sibling_id, sibling = siblings[index]
                    if sibling_id in seen:
                        continue
                    result.append(
                        {
                            "id": sibling_id,
                            "text": sibling["text"],
                            "meta": sibling["meta"],
                            "source": "context_neighbor",
                        }
                    )
                    seen.add(sibling_id)

        return result

    def build_context(self, passages: List[Dict]) -> str:
        """Build labelled, bounded context for the generation model."""
        parts = []
        total = 0
        source_no = 0

        for passage in passages:
            text = passage.get("text", "").strip()
            if not text:
                continue

            meta = passage.get("meta", {})
            source_no += 1
            source = meta.get("source", "okänd källa")
            page = meta.get("page")
            heading = meta.get("heading") or meta.get("section")

            location = []
            if page not in (None, ""):
                location.append(f"sida {page}")
            if heading:
                location.append(f"avsnitt {heading}")

            label = f"[KÄLLA {source_no}] {source}"
            if location:
                label += f" ({', '.join(location)})"

            block = f"{label}\n{text}"
            block_size = len(block) + 2
            if total + block_size > self.max_context_chars:
                break

            parts.append(block)
            total += block_size

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
        self,
        query: str,
        context: str,
        history: List[Dict],
    ) -> Generator[str, None, None]:
        system = (
            "Du är en spektrumexpert-assistent. Svara på svenska och använd endast "
            "informationen i KONTEXTEN som sakligt underlag. Om KONTEXTEN inte räcker "
            "för att besvara frågan, säg tydligt att underlaget inte räcker. Hitta inte "
            "på regler, frekvenser, dokument eller slutsatser. Skilj tydligt mellan vad "
            "källan säger och eventuella försiktiga slutsatser. Hänvisa till [KÄLLA n] "
            "när det är relevant."
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])
        messages.append(
            {
                "role": "user",
                "content": f"KONTEXT:\n\n{context}\n\nFRÅGA:\n{query}\n\nSVAR:",
            }
        )

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
