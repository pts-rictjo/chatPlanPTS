import streamlit as st
import pandas as pd
import chromadb
import ollama
import os
import subprocess
import time

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

desc_= [ "$ pip install streamlit pandas chromadb ollama",
"$ ollama pull qwen3",
"$ ollama pull llama3",
"$ ollama pull mxbai-embed-large",
"$ streamlit run query.py" ]

def pull_ollama_model(model_name: str):
    try:
        process = subprocess.run(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout

def get_local_ollama_models(retries=5, delay=1 , bOS = True ):
    if bOS :
        process		= subprocess.run( ['ollama', 'list'] ,
                              stdout = subprocess.PIPE,
                              stderr = subprocess.PIPE,
                              text   = True ,
                              check  = True )
        lines = process.stdout.splitlines()
        models = []
        for line in lines[1:]:  # hoppa över header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return sorted(models)
    else :
        for _ in range(retries):
            try:
                models = ollama.list()
                if models.get("models"):
                    return [m["name"] for m in models["models"]]
            except Exception:
                pass
            time.sleep(delay)
        return []


def is_chat_model(name: str) -> bool:
    return not any(
        kw in name.lower()
        for kw in ["embed", "embedding", "bge", "mxbai" , "name" ]
    )

st.set_page_config(page_title="Chatta med Excel (lokal LLM)", layout="wide")
st.title("Chatta med flera Excel-tabeller (Ollama)")

installed_models = get_local_ollama_models()
AVAILABLE_LLM_MODELS = [ m for m in installed_models if is_chat_model ( m ) ]


# Initiera Chroma (lokal vector DB)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="excel_data")


st.sidebar.header("Inställningar")

selected_llm = st.sidebar.selectbox(
    "Välj språkmodell (LLM)",
    AVAILABLE_LLM_MODELS,
    index=0
)

st.sidebar.markdown("### Lägg till ny modell")

new_model = st.sidebar.text_input(
    "Ange Ollama-modell (t.ex. gemma3, phi4 ... )",
    placeholder="modell:tag"
)

if st.sidebar.button("Hämta modell"):
    if not new_model.strip():
        st.sidebar.warning("Ange ett modellnamn.")
    else:
        with st.spinner(f"Hämtar {new_model}..."):
            success, output = pull_ollama_model(new_model.strip())

        if success:
            st.sidebar.success(f"Modellen '{new_model}' är nu installerad.")
            st.rerun() # VIKTIGT
        else:
            st.sidebar.error("Misslyckades att hämta modellen.")
            st.sidebar.code(output)

llm = selected_llm

st.info(f"Aktiv språkmodell: **{selected_llm}**")

# --- Filuppladdning ---
uploaded_files = st.file_uploader(f"Ladda upp en eller flera Excel-filer åt språkmodellen (modell: {selected_llm})" ,
			 type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Inlästa filer")
    for f in uploaded_files:
        st.write(f"➡ {f.name}")

    # --- Chunking-funktion ---
    def chunk_table(df, filename, max_rows=30):
        chunks = []
        for start in range(0, len(df), max_rows):
            chunk = df.iloc[start:start+max_rows]
            text = chunk.to_csv(index=False)
            chunks.append((filename, text))
        return chunks

    # --- Indexera alla filer i Chroma ---
    for file in uploaded_files:
        df = pd.read_excel(file)
        chunks = chunk_table(df, file.name)

        for i, (fname, ch) in enumerate(chunks):
            emb = ollama.embeddings(model="mxbai-embed-large", prompt=ch)["embedding"]
            collection.add(
                documents=[ch],
                embeddings=[emb],
                ids=[f"{fname}_chunk_{i}"],
                metadatas=[{"filename": fname}]
            )

    # --- Frågefunktion ---
    def ask_model(question):
        q_emb = ollama.embeddings(model="mxbai-embed-large", prompt=question)["embedding"]
        results = collection.query(query_embeddings=[q_emb], n_results=5)

        # Bygg kontext från flera filer
        context_blocks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context_blocks.append(f"(Fil: {meta['filename']})\n{doc}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
        Du är en frekvens spektrum hanterings assistent som svarar på frågor baserat på följande utdrag ur olika Excel-tabeller:

        {context}

        Fråga: {question}
        Svar:
        """

        response = ollama.chat(
            model = llm ,  # byt till önskad modell
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    # --- Inputfält för frågor ---
    st.subheader("Ställ en fråga över alla tabeller")
    user_question = st.text_input("Din fråga:")

    if user_question:
        with st.spinner("Tänker..."):
            answer = ask_model(user_question)
        st.success("Svar från modellen:")
        st.write(answer)
