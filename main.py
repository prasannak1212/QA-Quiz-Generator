from fastapi import FastAPI
from pydantic import BaseModel
from chains import qa_chain, quiz_chain

# -----------------------------
# Creating FastAPI instance
# -----------------------------
app = FastAPI()

# -----------------------------
# Pydantic Models
# -----------------------------
class QARequest(BaseModel):
    paragraph: str
    question: str


class QuizRequest(BaseModel):
    paragraph: str


# -----------------------------
# Creating QA endpoint
# -----------------------------
@app.post("/ask")
def ask_question(data: QARequest):
    response = qa_chain(
        context=data.paragraph,
        question=data.question
    )
    return {"answer": response}


# -----------------------------
# Creating Quiz endpoint
# -----------------------------
@app.post("/generate-quiz")
def generate_quiz(data: QuizRequest):
    response = quiz_chain(
        context=data.paragraph
    )
    return {"quiz": response}