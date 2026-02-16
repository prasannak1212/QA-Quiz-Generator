from fastapi import FastAPI
from pydantic import BaseModel
from chains import qa_chain, quiz_chain

app = FastAPI()

class QARequest(BaseModel):
    paragraph: str
    question: str

class QuizRequest(BaseModel):
    paragraph: str


@app.post("/ask")
def ask_question(data: QARequest):
    response = qa_chain.invoke({
        "context": data.paragraph,
        "question": data.question
    })
    return {"answer": response.content}


@app.post("/generate-quiz")
def generate_quiz(data: QuizRequest):
    response = quiz_chain.invoke({
        "context": data.paragraph
    })
    return {"quiz": response.content}
