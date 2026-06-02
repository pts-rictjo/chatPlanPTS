#!/bin/bash
set -e

echo "=== Chatplan Entrypoint ==="

# Vänta på Ollama om den körs lokalt (valfritt)
if [ -n "$WAIT_FOR_OLLAMA" ]; then
    echo "Väntar på Ollama på $OLLAMA_HOST ..."
    until curl -s "$OLLAMA_HOST" > /dev/null; do
        sleep 2
    done
    echo "Ollama svarar!"
fi

# Lista med modeller som behövs (anpassa efter dina faktiska modeller)
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

# Bygg ChromaDB om den inte redan finns
if [ ! -f /app/chroma_db/bm25.pkl ]; then
    echo "ChromaDB saknas – bygger från /app/data ..."
    python -c "
from helper.build_chroma_class import RAGSystem
rag = RAGSystem(data_root='/app/data', chroma_dir='/app/chroma_db')
rag.create_chromadb_from_data(table_as_document=True, group_tables=False)
rag.save_metadata()
"
    echo "ChromaDB skapad."
else
    echo "ChromaDB finns – hoppar över byggsteget."
fi

# Starta den tjänst som anges som CMD
exec "$@"
