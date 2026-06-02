#!/bin/bash
PORT=8000

echo "Kontrollerar om port $PORT används..."
# Döda process(er) på porten (fuser -k skickar SIGKILL)
sudo fuser -k $PORT/tcp 2>/dev/null

# Vänta så porten hinner frigöras
sleep 1

echo "Startar RAG-applikationen..."
python -m app.main
