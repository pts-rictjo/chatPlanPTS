from typing import Dict, List, Optional
import time


class Retriever:
    """Hybrid lexical + dense retriever using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        rag_system,
        dense_k: int = 30,
        bm25_k: int = 30,
        rrf_k: int = 30,
        rrf_constant: int = 60,
    ):
        self.rag = rag_system
        self.dense_k = max(1, dense_k)
        self.bm25_k = max(1, bm25_k)
        self.rrf_k = max(1, rrf_k)
        self.rrf_constant = max(1, rrf_constant)

    def dense_retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or self.dense_k
        q_emb = self.rag.generate_embedding(query)
        results = self.rag.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "id": meta.get("doc_id", chroma_id),
                "text": text,
                "meta": meta or {},
                "distance": distance,
                "source": "dense",
            }
            for chroma_id, text, meta, distance in zip(
                ids, docs, metas, distances
            )
        ]

    def bm25_retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        if getattr(self.rag, "bm25", None) is None:
            return []
        return self.rag.bm25_query(query, k=k or self.bm25_k)

    def hybrid_retrieve(self, query: str) -> List[Dict]:
        """Retrieve candidates from both indexes and fuse by rank.

        RRF deliberately does not combine raw Chroma distances with BM25
        scores: the two scores have different scales and semantics. Only rank
        is used for fusion.
        """
        t0 = time.perf_counter()

        dense = self.dense_retrieve(query, self.dense_k)
        t_dense = time.perf_counter()

        bm25 = self.bm25_retrieve(query, self.bm25_k)
        t_bm25 = time.perf_counter()

        rrf_scores: Dict[str, float] = {}
        docs: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                self.rrf_constant + rank
            )
            docs.setdefault(doc_id, doc)

        for rank, doc in enumerate(bm25, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                self.rrf_constant + rank
            )
            # Prefer the dense record when the same ID occurs in both lists,
            # because it retains the Chroma distance for diagnostics.
            docs.setdefault(doc_id, doc)

        merged = list(docs.values())
        for doc in merged:
            doc["rrf_score"] = rrf_scores.get(doc["id"], 0.0)

        merged.sort(key=lambda item: item["rrf_score"], reverse=True)
        result = merged[: self.rrf_k]

        if getattr(self.rag, "debug", False):
            print(
                f"[retrieval] dense={len(dense)} in {(t_dense-t0)*1000:.0f} ms, "
                f"bm25={len(bm25)} in {(t_bm25-t_dense)*1000:.0f} ms, "
                f"rrf={len(result)} in {(time.perf_counter()-t_bm25)*1000:.0f} ms"
            )

        return result
