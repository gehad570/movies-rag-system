# Movies RAG System

## Overview
This project uses Retrieval-Augmented Generation (RAG) to retrieve relevant movie information and answer user queries about movies.

It uses the TMDB 5000 Movies dataset, Sentence Transformers for text embeddings, ChromaDB for storing and retrieving movie information, and Qwen2.5-3B-Instruct for generating answers based on the retrieved context.

## Technologies Used
- Python
- Pandas
- Sentence Transformers
- ChromaDB
- Hugging Face Transformers
- Qwen2.5-3B-Instruct
- Streamlit (planned)

## Dataset
The project uses the TMDB 5000 Movies dataset, which contains movie titles, descriptions, and other movie-related information.

Dataset: https://huggingface.co/datasets/alejandrowallace/tmdb-5000

## How It Works
1. Load and preprocess the movie dataset.
2. Split movie descriptions into smaller text chunks.
3. Convert the chunks into embeddings using Sentence Transformers.
4. Store the embeddings and metadata in ChromaDB.
5. Retrieve relevant movie information based on the user's query.
6. Use Qwen2.5-3B-Instruct to generate an answer using the retrieved context.

## Example Queries
- What are some good movies to watch on a rainy day?
- Recommend movies about dreams and reality.
- Suggest movies with a psychological thriller storyline.

## Future Work
- Build a web interface using Streamlit.
- Improve retrieval quality and movie recommendations.

## Author
Gehad Fahdy
