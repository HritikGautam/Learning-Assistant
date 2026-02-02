from groq import Groq
import joblib
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# 1. Initialize the model (Must be the same model used in Step 4/5)
# This avoids the "localhost" error on Streamlit Cloud
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize Groq using Streamlit secrets
# Ensure you have 'gsk_r5...' set in your Streamlit dashboard secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def create_embedding(text_list):
    """
    Standardizes the embedding process to match the training data.
    text_list: list of strings
    returns: list of embedding vectors
    """
    # Use the local model instead of a requests.post to localhost
    embeddings = model.embed_documents(text_list)
    return embeddings


def inference(prompt):
    """
    Sends the formatted prompt (context + question) to Groq.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    # Return dictionary to match your app.py's ['response'] access
    return {"response": chat_completion.choices[0].message.content}


# Creating a embedding function for incoming query

embedding_cache = {}


def get_question_embedding(question):
    if question in embedding_cache:
        return embedding_cache[question]
    emb = create_embedding([question])[0]
    embedding_cache[question] = emb
    return emb


df = joblib.load("Step5_embeddings.joblib")
