import os
from pathlib import Path

import pandas as pd
import streamlit as st
import chromadb
import torch
from huggingface_hub import hf_hub_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
COLLECTION_NAME = "movies"
CHROMA_PATH = "./chroma_db"
CSV_FILENAME = "tmdb_5000_movies.csv"

st.set_page_config(
    page_title="Movies RAG System",
    page_icon="🎬",
    layout="centered",
)


# -----------------------------
# Load data and models
# -----------------------------
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    """Load a local CSV if available; otherwise download it from Hugging Face."""
    local_csv = Path("movies.csv")

    if local_csv.exists():
        df = pd.read_csv(local_csv)
    else:
        file_path = hf_hub_download(
            repo_id="alejandrowallace/tmdb-5000",
            filename=CSV_FILENAME,
            repo_type="dataset",
        )
        df = pd.read_csv(file_path)

    required_columns = {"original_title", "overview"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    df = df[["original_title", "overview"]].dropna().copy()
    df["original_title"] = df["original_title"].astype(str)
    df["overview"] = df["overview"].astype(str)
    df = df[df["overview"].str.strip().ne("")]
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading Qwen... This may take a few minutes.")
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


@st.cache_resource(show_spinner="Preparing the movie vector database...")
def load_or_build_collection():
    """Create a persistent ChromaDB collection and index movie chunks once."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        return collection

    df = load_movies()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )

    records = []
    for _, row in df.iterrows():
        for chunk in splitter.split_text(row["overview"]):
            if chunk.strip():
                records.append(
                    {
                        "title": row["original_title"],
                        "chunk": chunk,
                    }
                )

    if not records:
        raise ValueError("No movie descriptions were available to index.")

    embedder = load_embedding_model()
    batch_size = 64

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [item["chunk"] for item in batch]
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=[f"movie_chunk_{i}" for i in range(start, start + len(batch))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "original_title": item["title"],
                    "chunk": item["chunk"],
                }
                for item in batch
            ],
        )

    return collection


# -----------------------------
# RAG functions
# -----------------------------
def retrieve_movies(query: str, collection, embedder, top_k: int = 5):
    query_embedding = embedder.encode(query).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    retrieved = []
    for metadata, distance in zip(metadatas, distances):
        if metadata:
            retrieved.append(
                {
                    "title": metadata.get("original_title", "Unknown title"),
                    "chunk": metadata.get("chunk", ""),
                    "distance": distance,
                }
            )
    return retrieved


def generate_answer(query: str, retrieved_movies, generator):
    if not retrieved_movies:
        return "I couldn't find relevant movie information for that question."

    context = "\n\n".join(
        f"Movie: {item['title']}\nDescription: {item['chunk']}"
        for item in retrieved_movies
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation assistant. Answer using the movie "
                "information in the provided context. Recommend only movies present "
                "in the context, explain briefly why they may fit the request, and "
                "say when the context is insufficient. Do not invent plot details."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question: {query}\n\n"
                f"Retrieved movie context:\n{context}\n\n"
                "Give a clear, helpful answer with movie titles and short reasons."
            ),
        },
    ]

    tokenizer = generator.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = generator(
        prompt,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return output[0]["generated_text"].strip()


# -----------------------------
# Streamlit interface
# -----------------------------
st.title("🎬 Movies RAG System")
st.write(
    "Ask for movie recommendations using a theme, mood, or storyline. "
    "The system retrieves relevant movie descriptions and uses Qwen to form an answer."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Number of movie descriptions to retrieve",
        min_value=3,
        max_value=8,
        value=5,
    )
    st.caption("The first run downloads the dataset and builds the local vector database.")
    st.caption("Qwen2.5-3B may require substantial RAM or GPU memory.")

question = st.text_area(
    "What kind of movie are you looking for?",
    placeholder="e.g. Recommend movies about dreams and reality",
    height=100,
)

examples = [
    "What are some good movies to watch on a rainy day?",
    "Recommend movies about dreams and reality.",
    "Suggest a psychological thriller.",
]
st.write("Try an example:")
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state["movie_question"] = example

if "movie_question" in st.session_state and not question:
    question = st.session_state["movie_question"]

if st.button("Find movies", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Retrieving relevant movies..."):
                collection = load_or_build_collection()
                embedder = load_embedding_model()
                retrieved = retrieve_movies(question.strip(), collection, embedder, top_k)

            if not retrieved:
                st.info("No relevant movie descriptions were found.")
            else:
                with st.spinner("Generating recommendations with Qwen..."):
                    generator = load_generator()
                    answer = generate_answer(question.strip(), retrieved, generator)

                st.subheader("Recommendation")
                st.write(answer)

                with st.expander("View retrieved movie descriptions"):
                    for index, item in enumerate(retrieved, start=1):
                        st.markdown(f"**{index}. {item['title']}**")
                        st.write(item["chunk"])
                        st.divider()

        except Exception as exc:
            st.error(
                "The app could not complete the request. Check that the dependencies "
                "are installed and that your computer has enough memory to load Qwen."
            )
            st.exception(exc)

st.caption("Dataset: TMDB 5000 Movies. This project is for learning and demonstration.")
