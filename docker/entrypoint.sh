#!/bin/bash
set -e

echo "=== Chatplan Entrypoint / RAG index v2.1 ==="

if [ -n "$WAIT_FOR_OLLAMA" ]; then
    echo "Väntar på Ollama på $OLLAMA_HOST ..."
    until curl -s "$OLLAMA_HOST" > /dev/null; do
        sleep 2
    done
    echo "Ollama svarar!"
fi

sleep 3

REQUIRED_MODELS=(
    "mxbai-embed-large"
    "vanilj/llama-3.1-instruct-bellman-8b-swedish:q3_k_m"
)

for MODEL in "${REQUIRED_MODELS[@]}"; do
    if ! curl -s "$OLLAMA_HOST/api/tags" | grep -q "\"name\":\"$MODEL\""; then
        echo "Drar ner modellen $MODEL ..."
        curl -s -X POST "$OLLAMA_HOST/api/pull" -d "{\"name\":\"$MODEL\"}"
        echo "Klart: $MODEL"
    else
        echo "Modellen $MODEL finns redan."
    fi
done

INDEX_VERSION="$(python - <<'PY'
import json, os
p='/app/chroma_db/chromadb_params.json'
try:
    with open(p, encoding='utf-8') as f:
        print(json.load(f).get('INDEX_VERSION', 0))
except Exception:
    print(0)
PY
)"

if [ "$INDEX_VERSION" != "2" ]; then
    echo "Indexversion $INDEX_VERSION hittades – bygger om till RAG index v2."
    rm -rf /app/chroma_db
    mkdir -p /app/chroma_db
    python - <<'PY'
from helper.build_chroma_class import RAGSystem
rag = RAGSystem(data_root='/app/data', chroma_dir='/app/chroma_db')
rag.create_chromadb_from_data(table_as_document=True, group_tables=False)
rag.save_metadata()
PY
    echo "RAG index v2 skapad."
else
    echo "RAG index v2 finns – hoppar över omindexering."
fi

exec "$@"
