
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import chromadb
import torch

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------- Configuration --------------------

st.set_page_config(
    page_title="Movies RAG System",
    page_icon="🎬",
    layout="centered",
)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DB_PATH = "./chroma_db"
COLLECTION_NAME = "movies"
BATCH_SIZE = 64
TOP_K = 3


# -------------------- Text Chunking --------------------

def split_text(text, chunk_size=1000, chunk_overlap=20):
    """Split text into overlapping character chunks."""
    text = str(text).strip()

    if not text or text.lower() == "nan":
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + int(chunk_size * 0.7):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


# -------------------- Dataset --------------------

@st.cache_data(show_spinner=False)
def load_movies():
    local_file = "movies.csv"

    if os.path.exists(local_file):
        file_path = local_file
    else:
        file_path = hf_hub_download(
            repo_id="alejandrowallace/tmdb-5000",
            filename="tmdb_5000_movies.csv",
            repo_type="dataset",
        )

    df = pd.read_csv(file_path)

    required = {"original_title", "overview"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df = df[["original_title", "overview"]].copy()
    df = df.dropna(subset=["original_title", "overview"])

    df["original_title"] = df["original_title"].astype(str)
    df["overview"] = df["overview"].astype(str)

    df = df[
        (df["original_title"].str.strip() != "")
        & (df["overview"].str.strip() != "")
        & (df["overview"].str.lower() != "nan")
    ]

    return df.reset_index(drop=True)


# -------------------- Embedding Model --------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


# -------------------- ChromaDB --------------------

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    os.makedirs(DB_PATH, exist_ok=True)

    return chromadb.PersistentClient(path=DB_PATH)


def get_or_build_collection(progress):
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Reuse the existing database when it already has documents.
    if collection.count() > 0:
        progress.progress(1.0, text="Vector database ready.")
        return collection

    df = load_movies()
    embedding_model = load_embedding_model()

    records = []

    for _, row in df.iterrows():
        for chunk in split_text(row["overview"]):
            records.append({
                "title": row["original_title"],
                "chunk": chunk,
            })

    if not records:
        raise ValueError("No movie descriptions were found.")

    total = len(records)

    for start in range(0, total, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        texts = [item["chunk"] for item in batch]

        embeddings = embedding_model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        ids = [
            f"movie_chunk_{i}"
            for i in range(start, start + len(batch))
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"title": item["title"]}
                for item in batch
            ],
        )

        progress.progress(
            min((start + len(batch)) / total, 1.0),
            text=f"Building movie database: "
                 f"{min(start + len(batch), total)}/{total}",
        )

    return collection


# -------------------- Language Model --------------------

@st.cache_resource(show_spinner=False)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model.eval()

    return tokenizer, model


# -------------------- Retrieval --------------------

def retrieve_movies(question, collection, embedding_model):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for document, metadata in zip(documents, metadatas):
        retrieved.append({
            "title": metadata.get("title", "Unknown"),
            "overview": document,
        })

    return retrieved


# -------------------- Answer Generation --------------------

def generate_answer(question, retrieved, tokenizer, model):
    if not retrieved:
        return "I couldn't find relevant movie information in the dataset."

    context = "\n\n".join(
        f"Movie: {item['title']}\n"
        f"Overview: {item['overview']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie information assistant. "
                "Answer using only the movie information provided. "
                "If the context does not contain the answer, "
                "say that the available data is insufficient. "
                "Do not invent movie details. Keep answers concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Movie information:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer or "The model did not generate an answer. Please try again."


# -------------------- Streamlit Interface --------------------

st.title("🎬 Movies RAG System")

st.write(
    "Ask questions about movies using retrieval-augmented generation "
    "with the TMDB 5000 Movies dataset."
)

with st.sidebar:
    st.header("About")
    st.write("**Dataset:** TMDB 5000 Movies")
    st.write("**Retrieval:** Sentence Transformers + ChromaDB")
    st.write("**Generation:** Qwen2.5-1.5B-Instruct")

question = st.text_input(
    "Ask a movie question",
    placeholder="Recommend a movie about space exploration...",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Preparing the movie database..."):
                progress = st.progress(
                    0.0,
                    text="Checking the vector database...",
                )

                collection = get_or_build_collection(progress)
                progress.empty()

            with st.spinner("Loading the embedding model..."):
                embedding_model = load_embedding_model()

            with st.spinner("Searching for relevant movies..."):
                retrieved = retrieve_movies(
                    question,
                    collection,
                    embedding_model,
                )

            st.subheader("Retrieved Movies")

            if retrieved:
                for item in retrieved:
                    with st.expander(item["title"]):
                        st.write(item["overview"])
            else:
                st.info("No relevant movies were retrieved.")

            with st.spinner(
                "Loading the language model and generating your answer..."
            ):
                tokenizer, model = load_llm()

                answer = generate_answer(
                    question,
                    retrieved,
                    tokenizer,
                    model,
                )

            st.subheader("Answer")
            st.write(answer)

        except Exception as exc:
            st.error(
                "The app encountered an error while processing your question."
            )
            st.exception(exc)
