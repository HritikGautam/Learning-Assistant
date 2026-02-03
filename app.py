import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
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


embedding_cache = {}


def get_question_embedding(question):
    if question in embedding_cache:
        return embedding_cache[question]
    emb = create_embedding([question])[0]
    embedding_cache[question] = emb
    return emb


st.set_page_config(page_title="Video Q&A System", layout="wide")
st.title("Video to Q&A Pipeline(Ask Question About Video)")

st.header("You can ask questions about the following course:-")
st.markdown(
    "Visit the Course by clicking [here](https://youtube.com/playlist?list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w&si=P9ANYECyid5UBvyr)."
)
df = joblib.load("embeddings.joblib")

user_question = st.text_input("Ask your question:")

if user_question.strip():
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # Vector Search
            question_embedding = create_embedding([user_question])[0]
            similarities = cosine_similarity(
                np.vstack(df["embedding"]), [question_embedding]
            ).flatten()

            top_k = 3
            top_indices = similarities.argsort()[::-1][:top_k]
            top_chunks = df.loc[top_indices]

            # Build Prompt
            prompt = f""" Here are video subtitle chunks:
                        {top_chunks[["title", "number", "start", "end", "text"]].to_json(orient="records")}
                        ---------------------------------
                        User Question: "{user_question}"
                        If you are reffering to a video, also mention the video number like "Video number X". Where "X" will be the video number you are reffering to.
                        Also if you are providing timestamps, make sure you give it in "X:Y". Where X is the minute and Y is the second.
                        Answer in a human way and guide the user to exact video timestamps.
                        """

            # LLM Inference
            response = inference(prompt)["response"]
            st.markdown("### Answer")
            st.write(response)
