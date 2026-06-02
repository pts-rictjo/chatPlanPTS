from typing import List, Dict
import uuid
import time
import os

bDebug          = os.getenv("UI_DEBUG_RAGSERVICE", "false").lower() == "true"

class Retriever:
    def __init__(self, rag_system):
        self.rag = rag_system

    def dense_retrieve(self, query: str, k: int = 30) -> List[Dict]:
        """Dense retrieval via ChromaDB med query-embedding."""
        q_emb = self.rag.generate_embedding(query)
        results = self.rag.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents","metadatas","distances"] #
        )
        docs  = results['documents'][0]
        metas = results['metadatas'][0]
        return [
            {
                "id": m["doc_id"],
                "text": d,
                "meta": m,
                "score": 0.0,
                "source": "dense"
            }
            for d, m in zip(docs, metas)
        ]

    def bm25_retrieve(self, query: str, k: int = 30) -> List[Dict]:
        """BM25 retrieval om index finns."""
        if not hasattr(self.rag, 'bm25') or self.rag.bm25 is None:
            return []
        return self.rag.bm25_query(query, k=k)

    def hybrid_retrieve(self, query: str, k_dense: int = 30, k_bm25: int = 30, top_k: int = 30) -> List[Dict]:
        """
        Kombinera dense och BM25 med reciprocal rank fusion (RRF).
        """
        if bDebug :
            print(f"[DEBUG] Hybrid retrieve for query: {query[:50]}...")
            t0 = time.time()
        dense_results = self.dense_retrieve(query, k=k_dense)
        if bDebug :
            print(f"[DEBUG] Dense got {len(dense_results)} docs in {time.time()-t0:.2f}s")
            t1 = time.time()
        bm25_results = self.bm25_retrieve(query, k=k_bm25)
        if bDebug :
            print(f"[DEBUG] BM25 got {len(bm25_results)} docs in {time.time()-t1:.2f}s")

        # RRF-skoring
        scores = {}
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
            doc["rrf_score"] = scores[doc_id]

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank)
            doc["rrf_score"] = scores[doc_id]

        # Sammanfoga alla unika dokument
        all_docs = {doc["id"]: doc for doc in dense_results + bm25_results}
        merged = list(all_docs.values())
        merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return merged[:top_k]
