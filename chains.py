import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# LLM Config
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

# -----------------------------
# Embedding Model (local)
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Utility: Split text into chunks
# -----------------------------
def split_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Build FAISS index
# -----------------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    # embeddings = embedder.encode(chunks, normalize_embeddings=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve(query, chunks, index, top_k=2, threshold=0.8):
    query_embedding = embedder.encode([query])
    # index = faiss.IndexFlatIP(dimension)
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    retrieved_chunks = []

    for i, distance in zip(indices[0], distances[0]):

        if distance < threshold:   # lower distance = more similar
            retrieved_chunks.append(chunks[i])

    return retrieved_chunks


# -----------------------------
# Call Ollama
# -----------------------------
def generate_response(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# -----------------------------
# QA Chain (Now RAG-based)
# -----------------------------
def qa_chain(context: str, question: str):
    chunks = split_text(context)
    index, _ = build_faiss_index(chunks)
    retrieved_chunks = retrieve(question, chunks, index)

    final_context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question strictly using the retrieved context below.
If answer not present, say "Answer not found in paragraph."

Retrieved Context:
{final_context}

Question:
{question}

Answer:
"""
    if not retrieved_chunks:
        return "Answer not found in paragraph."
    # else:
    return generate_response(prompt)


# -----------------------------
# Quiz Chain (Now RAG-based)
# -----------------------------
def quiz_chain(context: str):
    chunks = split_text(context)
    index, _ = build_faiss_index(chunks)

    # For quiz, retrieve top 3 chunks using a generic query
    retrieved_chunks = retrieve("Generate quiz from this text", chunks, index, top_k=3)

    final_context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Generate 5 quiz questions strictly based on the context below.

Context:
{final_context}

Quiz Questions:
1.
2.
3.
4.
5.
"""

    return generate_response(prompt)