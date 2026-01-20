# Ladda ned Docker bilden här :
https://hub.docker.com/r/rictjo/chatplan
```
docker pull rictjo/chatplan:freq
```
# Lokal Excel LLM App

Detta projekt tillhandahåller en **självbärande Streamlit-applikation** som låter användare ställa frågor över en eller flera Excel-filer med hjälp av **lokala LLM-modeller via Ollama**.

Projektet är designat för att fungera **plattformoberoende**:

* Linux (CPU, AMD GPU via ROCm/MESA)
* Windows 11 (CPU eller NVIDIA GPU via Docker Desktop)
* macOS (CPU)

Samma Docker-image används i alla fall – skillnaden ligger endast i **hur containern startas**.

---

## Arkitektur – översikt

* **Streamlit** – webbgränssnitt (`http://localhost:8501`)
* **Ollama** – lokal LLM-server (CPU/GPU automatiskt)
* **ChromaDB** – lokal vektordatabas
* **Docker** – distribution och isolering

GPU-stöd aktiveras **vid runtime**, inte i koden.

---

## Projektstruktur

```
excel-llm-app/
├── app/
│   └── query_llm.py
├── docker/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── compose.cpu.yml
│   ├── compose.nvidia.yml
│   └── compose.amd.yml
├── requirements.txt
└── README.md
```

---

## Hur man bygger docker-imagen

Byggs en gång (på Linux eller Windows):

```bash
docker build -t excel-llm-app -f docker/Dockerfile .
```

---

## Starta applikationen

### CPU-only (alla plattformar)

```bash
docker compose -f docker/compose.cpu.yml up
```

### NVIDIA GPU (Windows eller Linux)

Krav:

* NVIDIA GPU
* NVIDIA-drivrutin
* Docker Desktop med GPU-stöd

```bash
docker compose -f docker/compose.nvidia.yml up
```

Alternativt:

```bash
docker run --gpus all -p 8501:8501 -v ollama-data:/root/.ollama excel-llm-app
```

### AMD GPU (Linux / NixOS)

Krav:

* AMD GPU
* ROCm/MESA installerat på host

```bash
docker compose -f docker/compose.amd.yml up
```

MD GPU stöds **inte** på Windows i nuläget – där används CPU fallback.

---

## Användning

1. Starta containern (enligt ovan)
2. Öppna webbläsare:

```
http://localhost:8501
```

3. Ladda upp en eller flera `.xlsx`-filer
4. Välj språkmodell (endast installerade Ollama-modeller visas)
5. Ställ frågor över samtliga tabeller

---

## Modeller ("thin image")

Docker-imagen innehåller **inga LLM-modeller** i sin första start.

* Modeller laddas automatiskt via Ollama vid första användning
* Laddade modeller sparas i Docker-volymen:

```
ollama-data → /root/.ollama
```

Nästa start är omedelbar.

---

## Vanliga frågor

### Behöver användaren installera Python eller Ollama?

Nej – allt körs i Docker.

### Fungerar detta offline?

Ja, efter att modeller laddats första gången.

### Hur stor är imagen?

* Basimage: ~1–2 GB
* Modeller: 2–40+ GB beroende på val

---

## Begränsningar

* AMD GPU stöds endast på Linux
* Windows + AMD → CPU-only
* Mycket stora modeller kräver mycket RAM/VRAM

---

## Licenser

Detta projekt är Apache-2 men avsett för intern användning, experiment och forskning.
Respektera respektive modells licensvillkor (Gemma, LLaMA, m.fl.).

---

## Rekommenderad vidareutveckling

* Multi-user sessioner
* Persistens av ChromaDB

---

**Bygg en image. Kör den överallt. Låt Ollama göra resten.** 🚀
