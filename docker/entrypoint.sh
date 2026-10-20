#!/bin/bash
set -e

echo "=== Chatplan Entrypoint ==="

# ------------------------------------------------------------
# Vänta på Ollama
# ------------------------------------------------------------
if [ -n "$WAIT_FOR_OLLAMA" ]; then
    echo "Väntar på Ollama på $OLLAMA_HOST ..."

    until curl -fsS "$OLLAMA_HOST" > /dev/null; do
        sleep 2
    done

    echo "Ollama svarar!"
fi

sleep 3

# ------------------------------------------------------------
# Kontrollera nödvändiga modeller
# ------------------------------------------------------------
REQUIRED_MODELS=(
    "mxbai-embed-large"
    "vanilj/llama-3.1-instruct-bellman-8b-swedish:q3_k_m"
)

for MODEL in "${REQUIRED_MODELS[@]}"; do

    if ! curl -fsS "$OLLAMA_HOST/api/tags" | grep -q "\"name\":\"$MODEL\""; then
        echo "Modellen saknas: $MODEL"
        echo "Drar ner modellen ..."

        curl -fsS \
            -X POST \
            "$OLLAMA_HOST/api/pull" \
            -d "{\"name\":\"$MODEL\"}"

        echo
        echo "Klart: $MODEL"
    else
        echo "Modellen finns redan: $MODEL"
    fi

done

# ------------------------------------------------------------
# ChromaDB / indexversion
# ------------------------------------------------------------
INDEX_VERSION_REQUIRED=3
INDEX_PARAMS="/app/chroma_db/chromadb_params.json"

REBUILD_INDEX=false
INDEX_VERSION=0

if [ ! -d "/app/chroma_db" ]; then

    echo "ChromaDB-katalog saknas."
    REBUILD_INDEX=true

elif [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then

    echo "ChromaDB-katalogen är tom."
    REBUILD_INDEX=true

elif [ ! -f "$INDEX_PARAMS" ]; then

    echo "Ingen chromadb_params.json hittades."
    echo "Indexet betraktas som gammalt."
    REBUILD_INDEX=true

else

    INDEX_VERSION=$(python -c "
import json

try:
    with open('$INDEX_PARAMS', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(data.get('INDEX_VERSION', 0))

except Exception:
    print(0)
")

    echo "Befintlig ChromaDB indexversion: $INDEX_VERSION"
    echo "Krävd indexversion: $INDEX_VERSION_REQUIRED"

    if [ "$INDEX_VERSION" -lt "$INDEX_VERSION_REQUIRED" ]; then
        echo "Indexet är äldre än aktuell version."
        REBUILD_INDEX=true
    fi

fi

# ------------------------------------------------------------
# Bygg om ChromaDB vid behov
# ------------------------------------------------------------
if [ "$REBUILD_INDEX" = true ]; then

    echo "=============================================="
    echo "Bygger ChromaDB index version $INDEX_VERSION_REQUIRED"
    echo "Källa: /app/data"
    echo "=============================================="

    rm -rf /app/chroma_db
    mkdir -p /app/chroma_db

    python -c "
from helper.build_chroma_class import RAGSystem

rag = RAGSystem(
    data_root='/app/data',
    chroma_dir='/app/chroma_db'
)

rag.create_chromadb_from_data(
    table_as_document=True,
    group_tables=False
)

rag.save_metadata()
"

    echo "=============================================="
    echo "ChromaDB skapad."
    echo "=============================================="

else

    echo "ChromaDB indexversion $INDEX_VERSION är aktuell."
    echo "Hoppar över indexering."

fi

# ------------------------------------------------------------
# Starta applikationen
# ------------------------------------------------------------
echo "Startar Chatplan..."

exec "$@"
