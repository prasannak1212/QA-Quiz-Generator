Need to create a Q&A and Quiz generator RAG Application.

Tech Used:
API Layer - FASTAPI
Frontend - STREAMLIT
RAG Layer - Chunking + Embedding (sentence_transformers) + Vector Search (FAISS)
LLM Layer - Local Ollama phi3:mini model

System flow:
User enter text paragraph as input and select endpoint
streamlit UI sends request
Appropriate FastAPI endpoint catches the HTTP request
RAG Layer - divide the input into chunks + convert them into embeddings + Store the final vectors as FAISS index files
Convert the user query into embeddings
Perform Similarity search between the query embeddings and vector stores
Get appropriate chunks and construct a prompt with context
Call LLM to generate answer
FastAPI creates response with the generated answer
Answer display in streamlit UI

Architecture:
user ----> Streamlit ----> FastAPI endpoint ----> RAG layer ----> LLM generator ----> FastAPI response ----> Streamlit ----> User

