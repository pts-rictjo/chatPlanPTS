import streamlit as st

st.set_page_config(page_title="Chatta med Excel (lokal LLM)", layout="wide")
st.title("Chatta med flera Excel-tabeller (Ollama)")

import pandas as pd
import chromadb
import ollama

# Initiera Chroma (lokal vector DB)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="excel_data")

desc_= [ "$ pip install streamlit pandas chromadb ollama",
"$ ollama pull llama3",
"$ ollama pull mxbai-embed-large",
"$ streamlit run query.py" ]

# --- Filuppladdning ---
uploaded_files = st.file_uploader("Ladda upp en eller flera Excel-filer", type=["xlsx"], accept_multiple_files=True)

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
        Du är en assistent som svarar på frågor baserat på följande utdrag ur olika Excel-tabeller:

        {context}

        Fråga: {question}
        Svar:
        """

        response = ollama.chat(
            model="llama3",  # byt till önskad modell
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
