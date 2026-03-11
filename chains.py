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
# Embedding Model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Split text into chunks
# -----------------------------
def split_text(text, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Build FAISS index (cosine similarity)
# -----------------------------
def build_faiss_index(chunks):

    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)   # cosine similarity. It is vector DB object that can store vectors of dimension
    index.add(np.array(embeddings).astype("float32"))

    return index


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve(query, chunks, index, top_k=2, threshold=0.3):

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    retrieved_chunks = []

    for idx, score in zip(indices[0], scores[0]):

        if score > threshold:
            retrieved_chunks.append(chunks[idx])

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
# Validate Paragraph length
# -----------------------------
def limit_context(text, max_words=2000):
    words = text.split()

    if len(words) > max_words:
        words = words[:max_words]

    return " ".join(words)

# -----------------------------
# QA Chain (RAG)
# -----------------------------
def qa_chain(context: str, question: str):

    chunks = split_text(context)

    index = build_faiss_index(chunks)

    retrieved_chunks = retrieve(question, chunks, index)

    if not retrieved_chunks:
        return "Answer not found in paragraph."

    final_context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a question answering system.

    Rules:
    - Only answer using the provided context.
    - Do NOT use outside knowledge.
    - If the answer is not explicitly stated in the context, reply exactly:
    "Answer not found in paragraph."

    Context:
    {final_context}

    Question:
    {question}

    Answer:
"""

    return generate_response(prompt)


# -----------------------------
# Quiz Chain (RAG)
# -----------------------------
def quiz_chain(context: str):

    chunks = split_text(context)

    final_context = "\n\n".join(chunks)

    # Prevent context overflow
    final_context = limit_context(final_context)

    prompt = f"""
ROLE: You are a quiz generating system

RULES:
1. Generate strictly MCQ type questions from the context
2. Do NOT use outside knowledge
3. Strictly use the paragraph to generate quiz
4. Do NOT hallucinate, if not sure reply: "Unable to create Quiz"
5. Create exactly 5 questions each with 4 options

CONTEXT:
{final_context}

QUESTIONS:
1.
2.
3.
4.
5.
"""

    return generate_response(prompt)