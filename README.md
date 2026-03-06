# 📘 Paragraph Q&A + Quiz Generator (Local RAG + Ollama)

A **local Retrieval-Augmented Generation (RAG)** application that allows users to ask questions or generate quizzes from a paragraph.
The system retrieves relevant context using **FAISS vector search** and generates answers using a **local LLM (phi3-mini via Ollama)**.

This project demonstrates a **complete GenAI architecture** with:

* Frontend UI (Streamlit)
* Backend API (FastAPI)
* Retrieval-Augmented Generation (FAISS + Embeddings)
* Local LLM inference (Ollama)

---

# 🚀 Features

* Ask questions based on a paragraph
* Generate quiz questions from the paragraph
* Local AI inference (no external API)
* Retrieval-Augmented Generation using FAISS
* Embedding-based semantic search
* Hallucination reduction using similarity thresholds
* Simple and modular architecture

---

# 🏗️ System Architecture

```
User
  ↓
Streamlit Frontend
  ↓ HTTP request
FastAPI API Layer
  ↓
RAG Layer
   ├─ Text Chunking
   ├─ Sentence Embeddings
   ├─ FAISS Vector Search
   ├─ Context Retrieval
  ↓
LLM Layer (phi3-mini via Ollama)
  ↓
Generated Answer / Quiz
```

---

# 🧠 How the RAG Pipeline Works

1. **User inputs a paragraph and question**

2. **Text Chunking**

   * The paragraph is split into smaller chunks for better retrieval.

3. **Embeddings Generation**

   * Each chunk is converted into vector embeddings using:

```
all-MiniLM-L6-v2
```

4. **Vector Indexing**

   * The embeddings are stored in a **FAISS index** for efficient similarity search.

5. **Query Embedding**

   * The user's question is converted into an embedding.

6. **Similarity Search**

   * FAISS retrieves the most relevant chunks based on cosine similarity.

7. **Context Filtering**

   * Only chunks with similarity above a threshold are used.

8. **Prompt Construction**

   * Retrieved context + question are sent to the LLM.

9. **LLM Response**

   * The local **phi3-mini model** generates the answer.

---

# 🧩 Project Structure

```
project-folder/
│
├── frontend.py        # Streamlit UI
├── main.py            # FastAPI API layer
├── chains.py          # RAG + LLM logic
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

| Component     | Technology            |
| ------------- | --------------------- |
| Frontend      | Streamlit             |
| Backend API   | FastAPI               |
| Vector Search | FAISS                 |
| Embeddings    | Sentence Transformers |
| LLM           | phi3-mini             |
| LLM Runtime   | Ollama                |
| Language      | Python                |

---

# 🖥️ Installation

## 1️⃣ Clone Repository

```
git clone https://github.com/prasannak1212/QA-Quiz-Generator.git
cd QA-Quiz-Generator
```

---

## 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install fastapi
pip install uvicorn
pip install streamlit
pip install requests
pip install faiss-cpu
pip install sentence-transformers
pip install numpy
```

---

## 4️⃣ Install Ollama

Download from:

```
https://ollama.com
```

---

## 5️⃣ Pull the LLM Model

```
ollama pull phi3:mini
```

---

# ▶️ Running the Application

## Step 1: Start Ollama

```
ollama serve
```

---

## Step 2: Start FastAPI Backend

```
uvicorn main:app --reload
```

API will run at:

```
http://localhost:8000
```

---

## Step 3: Start Streamlit Frontend

```
streamlit run frontend.py
```

Frontend will open at:

```
http://localhost:8501
```

---

# 📚 Example Usage

### Paragraph

```
The history of India began with the Indus Valley Civilization.
Later periods included the Vedic age, Mauryan Empire, and Gupta Empire.
These empires contributed greatly to science, art, and literature.
```

### Example Question

```
Which empire contributed to science and literature?
```

### Example Output

```
The Mauryan and Gupta empires contributed significantly to science, art, and literature.
```

---

# 🛡️ Hallucination Control

The system reduces hallucinations using:

* Vector similarity threshold filtering
* Context-restricted prompts
* Retrieval-based grounding

If the answer is not found in the paragraph, the system returns:

```
Answer not found in paragraph.
```

---

# 📊 Future Improvements

Possible enhancements:

* Persistent vector database (Chroma / Pinecone)
* Multi-document support
* PDF / document upload
* Streaming LLM responses
* Model selection
* Docker deployment
* Cloud deployment (AWS / HuggingFace)

---

# 🎯 Learning Objectives

This project demonstrates:

* Building a **local GenAI application**
* Implementing **Retrieval-Augmented Generation**
* Using **vector embeddings and similarity search**
* Integrating **FastAPI + Streamlit + LLM**
* Running **LLMs locally with Ollama**

---

# 👨‍💻 Author

Built as a learning project to understand **RAG systems and local LLM deployment**.
