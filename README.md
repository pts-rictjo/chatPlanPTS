# Design
Det nya projektet har denna layout
```
project/
├── helper/
│   └── build_chroma_class.py   ← befintlig chromaDB klass
├── app/
│   ├── __init__.py
│   ├── main.py                  ← Kör FastAPI + statiska filer
│   ├── rag_service.py           ← RAG-flöde (retrieve, rerank, context)
│   ├── retriever.py             ← Hybridretriever (dense + BM25)
│   └── conversation_store.py    ← konversationshistorik
├── static/
│   └── index.html               ← frontend-chatt
│   └── spektrum_explorer.html   ← frontend-utforskare
├── public/                      ← publik media
├── data/                        ← originaldokument
├── chroma_db/                   ← ChromaDB och BM25.pkl
├── env/                         ← NixOS Linux miljö
├── docker/                      ← Docker miljö
└── requirements.txt
```
# Lokal testning
Gå in i nix miljön
```
nix-shell env/shell.nix
```

# Starta Ollama (sidecar):
```
ollama serve
```
(om den inte redan körs) och hämta din modell, t.ex. ollama pull llama3.

# Bygg databasen
Flytta dokument till data och kör

```
python helper/build_chroma_class.py
```

# Kör FastAPI:
```
python -m app.main
```
från projektroten (där app/ och helper/ ligger).

# Öppna 
```
http://localhost:8000
```
En fullständig RAG-chatt med RAGSystem som backend borde vara tillgängligt.


# Docker

## Skapa
```
docker build -t rictjo/chatplan:freq -f docker/Dockerfile .
```

## Kör

```
docker run \
  --network host \
  -e OLLAMA_HOST=http://127.0.0.1:11434 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  rictjo/chatplan:freq
```
