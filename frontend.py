import streamlit as st
import requests

st.title("📘 Paragraph Q&A + Quiz Generator")

# Paragraph Input
paragraph = st.text_area("Enter Paragraph", height=200)

# Mode Selection
mode = st.radio("Select Mode", ["Ask Question", "Generate Quiz"])

# ---------------------------
# Ask Question Mode
# ---------------------------

if mode == "Ask Question":
    question = st.text_input("Enter Question")

    if st.button("Ask"):
        response = requests.post(
            "http://localhost:8000/ask",
            json={
                "paragraph": paragraph,
                "question": question
            }
        )

        st.write(response.json()["answer"])

# ---------------------------
# Generate Quiz Mode
# ---------------------------

elif mode == "Generate Quiz":
    if st.button("Generate Quiz"):
        response = requests.post(
            "http://localhost:8000/generate-quiz",
            json={
                "paragraph": paragraph
            }
        )

        st.write(response.json()["quiz"])
