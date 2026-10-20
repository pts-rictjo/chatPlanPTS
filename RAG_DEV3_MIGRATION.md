# ChatPlan RAG v3 retrieval integration

These four files are intended to replace the corresponding files in the existing project:

- `helper/build_chroma_class.py`
- `app/retriever.py`
- `app/rag_service.py`
- `app/main.py`

No frontend changes are required. `app/conversation_store.py` is unchanged.

## Retrieval pipeline

```text
question
  -> BM25 top 30 + dense top 30
  -> RRF top 30
  -> local FlashRank top 7
  -> neighbour/context expansion
  -> LLM
```

The reranker is local FlashRank; there is **no Cohere dependency or API call**.

## Recommended environment

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

Set `UI_USE_RANKER_MODEL=false` to compare RRF-only retrieval.

## Important: rebuild the index

`build_chroma_class.py` now uses `INDEX_VERSION = 3`.

The index must be rebuilt because:

1. PDF prose is stored as bounded section/page chunks instead of whole-document text.
2. PDF chunks use a document-level `parent_id` and global `chunk_index`, allowing neighbour expansion across page boundaries.
3. Text deduplication is by stable chunk ID rather than text hash.
4. BM25 is built from exactly the same validated records that are stored in Chroma.
5. `group_tables` defaults to `False`, so table/text metadata is not destroyed by a second grouping step.

The existing `chroma_db/` should therefore be deleted/rebuilt, or the existing deployment's index-build mechanism should be made to rebuild when `INDEX_VERSION != 3`.

For a manual rebuild from the project root:

```bash
python -c "from helper.build_chroma_class import RAGSystem; r=RAGSystem(data_root='./data', chroma_dir='./chroma_db'); r.create_chromadb_from_data(table_as_document=True, group_tables=False); r.save_metadata()"
```

The rebuild requires Ollama to be available with the embedding model `mxbai-embed-large`.

## FlashRank

FlashRank was already part of the previous application. No Cohere package or API key is needed.

The application starts even if FlashRank cannot be initialized; in that case it logs the error and falls back to RRF results. This makes deployment failure less severe, but the intended production configuration is to have FlashRank installed and enabled.

## Diagnostics

With:

```text
UI_DEBUG_RAGSERVICE=true
```

the backend logs retrieval and reranking timings, for example:

```text
[retrieval] dense=30 in ... ms, bm25=30 in ... ms, rrf=30 in ... ms
[rag] retrieval=... ms, rerank=... ms, context=... ms, candidates=30, final=..., context_chars=...
```

This makes it possible to separate indexing/retrieval problems from LLM generation latency.
