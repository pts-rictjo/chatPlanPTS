# ChatPlan RAG v2.1 – migration

## Pipeline

```text
question
   |
   +--> BM25 top 30 ----+
   |                     |
   +--> dense top 30 ----+--> RRF top 30 --> FlashRank top 7
                                                   |
                                                   v
                                           neighbour expansion
                                                   |
                                                   v
                                                 LLM
```

## What changed from v2

v2.1 removes Cohere completely. The reranker is the existing local FlashRank
implementation used by the previous application:

```text
ms-marco-MiniLM-L-12-v2
```

No API key, external reranking service or network call is required for reranking.

The retrieval pipeline remains:

1. Dense retrieval with ChromaDB + Ollama embeddings.
2. BM25 retrieval with the existing `rank_bm25` index.
3. Reciprocal Rank Fusion (RRF).
4. Local FlashRank reranking.
5. Neighbour/context expansion.
6. Ollama LLM generation.

## Environment

Recommended settings:

```text
UI_USE_RANKER_MODEL=true
UI_RANKER_MODEL=ms-marco-MiniLM-L-12-v2
UI_DENSE_K=30
UI_BM25_K=30
UI_RRF_K=30
UI_RERANK_K=30
UI_TOP_K=7
UI_CONTEXT_NEIGHBORS=1
UI_MAX_CONTEXT_CHARS=14000
```

For a diagnostic comparison without reranking:

```text
UI_USE_RANKER_MODEL=false
```

The system then uses RRF directly.

## Important: rebuild the index

The index must be rebuilt. The previous PDF path stored the whole extracted PDF
text as one Chroma document. RAG v2 stores page/section-aware chunks with stable
IDs and metadata. The entrypoint automatically rebuilds when `INDEX_VERSION != 2`.

## Files

Replace the corresponding application files with:

- `build_chroma_class.py` -> `helper/build_chroma_class.py`
- `retriever.py` -> `app/retriever.py`
- `rag_service.py` -> `app/rag_service.py`
- `main.py` -> `app/main.py`
- `entrypoint.sh` -> deployment entrypoint

No new Python dependency is introduced by v2.1. FlashRank must already be present
as it was in the previous application.
