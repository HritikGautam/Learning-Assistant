import streamlit as st
import os

# import subprocess
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import Step1_process_video
import Step2_mp3_to_json
import Step3_merge_chunks
import Step4_preprocess_json

# import Step6_process_incoming
from Step6_process_incoming import create_embedding, inference

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "processing" not in st.session_state:
    st.session_state.processing = False


st.set_page_config(page_title="Video Q&A System", layout="wide")
st.title("🎥 Video to Q&A Pipeline")

# Folders
os.makedirs("videos", exist_ok=True)
os.makedirs("audios", exist_ok=True)
os.makedirs("jsons", exist_ok=True)
os.makedirs("merged_jsons", exist_ok=True)

# ---------------- VIDEO UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mkv", "avi"],
    disabled=st.session_state.processing,
)

if uploaded_file:
    # delete old video if exists
    video_path = os.path.join("videos", uploaded_file.name)

    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        os.remove(st.session_state.video_path)

    # Only write if file doesn't already exist
    if not os.path.exists(video_path):
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Video saved as {uploaded_file.name}")
    else:
        st.info(f"Video {uploaded_file.name} already exists. Using existing file.")
    st.session_state.video_uploaded = True
    st.session_state.video_path = video_path


# ---------------- PROCESS BUTTON ----------------
if st.button("Process Video"):
    if not st.session_state.video_uploaded:
        st.warning("Please upload a video first.")
    else:
        st.session_state.processing = True
        with st.spinner("Processing video... This may take a while."):
            Step1_process_video.run()
            Step2_mp3_to_json.run()
            Step3_merge_chunks.run()
            # print("Creating Embeddings...")
            Step4_preprocess_json.run()

        st.session_state.processing = False
        st.success("Processing completed! You can ask questions now.")
        # Step6_process_incoming.run()
        # st.success("Pipeline completed! You can now ask questions.")

# ---------------- QUESTION ANSWERING ----------------
st.header("Ask Question About Video")

can_answer = (
    st.session_state.video_uploaded
    and not st.session_state.processing
    and os.path.exists("Step5_embeddings.joblib")
)

# Load embeddings only if file exists
df = None
if can_answer:
    df = joblib.load("Step5_embeddings.joblib")
    # ✅ Check if 'embedding' column exists
    if "embedding" not in df.columns:
        # st.warning("Embeddings not found. Please process the video first.")
        can_answer = False

if not st.session_state.video_uploaded:
    st.info("Upload and process a video to ask questions.")

elif st.session_state.processing:
    st.warning("Video is still processing. Please wait.")

elif not os.path.exists("Step5_embeddings.joblib"):
    st.warning("Please process the video first.")


else:
    df = joblib.load("Step5_embeddings.joblib")
    user_question = st.text_input("Ask your question:", disabled=not can_answer)
    if st.button("Get Answer", disabled=not can_answer):
        if not user_question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Thinking..."):
                question_embedding = create_embedding([user_question])[0]
                similarities = cosine_similarity(
                    np.vstack(df["embedding"]), [question_embedding]
                ).flatten()
                top_k = 3
                top_indices = similarities.argsort()[::-1][:top_k]
                top_chunks = df.loc[top_indices]

                prompt = f""" Here are video subtitle chunks:

{top_chunks[["start", "end", "text"]].to_json(orient="records")}

---------------------------------
User Question:
"{user_question}"

Answer in a human way and guide the user to exact video timestamps.
"""

            response = inference(prompt)["response"]
            st.markdown("### Answer")
            st.write(response)
# else:
#     st.warning("Please process a video first.")
