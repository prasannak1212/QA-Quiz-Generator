import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace

# Load environment variables
load_dotenv()

# Create HuggingFace Chat model (router compatible)
llm = ChatHuggingFace(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=512,
)

# ---------------------------
# Q&A Prompt
# ---------------------------
qa_prompt = ChatPromptTemplate.from_template(
    """
Answer the question strictly based only on the given paragraph.
If answer not present, say "Answer not found in paragraph."

Paragraph:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = qa_prompt | llm


# ---------------------------
# Quiz Prompt
# ---------------------------
quiz_prompt = ChatPromptTemplate.from_template(
    """
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
)

quiz_chain = quiz_prompt | llm
