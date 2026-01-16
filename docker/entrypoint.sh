#!/usr/bin/env bash
set -e

echo "Startar Ollama..."
ollama serve &

until ollama list >/dev/null 2>&1; do
  sleep 1
done

ollama pull gemma3
ollama pull mxbai-embed-large

echo "Startar Streamlit..."
streamlit run query_llm.py \
  --server.address=0.0.0.0 \
  --server.port=8501
