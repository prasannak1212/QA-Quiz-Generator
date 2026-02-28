import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"


def generate_response(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def qa_chain(context: str, question: str):
    prompt = f"""
Answer the question strictly based only on the given paragraph.
If answer not present, say "Answer not found in paragraph."

Paragraph:
{context}

Question:
{question}

Answer:
"""
    return generate_response(prompt)


def quiz_chain(context: str):
    prompt = f"""
Generate 5 quiz questions based only on the paragraph below.

Paragraph:
{context}

Quiz Questions:
1.
2.
3.
4.
5.
"""
    return generate_response(prompt)