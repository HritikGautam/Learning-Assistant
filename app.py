import streamlit as st
import os
import shutil
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import Step1_process_video
import Step2_mp3_to_json
import Step3_merge_chunks
import Step4_preprocess_json
from Step6_process_incoming import create_embedding, inference


# --- 1. CLEANUP UTILITY ---
def cleanup_workspace():
    """Removes all temporary processing files and folders to avoid data overlap."""
    folders = ["audios", "jsons", "merged_jsons", "videos"]
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Remove the old embeddings database
    if os.path.exists("Step5_embeddings.joblib"):
        os.remove("Step5_embeddings.joblib")

    # Optional: Clear search results from session state
    if "user_question" in st.session_state:
        st.session_state.user_question = ""


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
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mkv", "avi"])

if uploaded_file:
    # SIMPLEST FIX: If the file name is different, delete the old database immediately
    if st.session_state.get("current_video") != uploaded_file.name:
        if os.path.exists("Step5_embeddings.joblib"):
            os.remove("Step5_embeddings.joblib")
        st.session_state.current_video = uploaded_file.name
        st.session_state.video_uploaded = True

    # Save the file
    video_path = os.path.join("videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_path = video_path


# ---------------- PROCESS BUTTON ----------------
if st.button("Process Video"):
    if not st.session_state.video_uploaded:
        st.warning("Please upload a video first.")
    else:
        # Extra cleanup right before processing starts to ensure fresh folders
        cleanup_workspace()

        st.session_state.processing = True
        with st.spinner("Processing video... This may take a while."):
            Step1_process_video.run()
            Step2_mp3_to_json.run()
            Step3_merge_chunks.run()
            # print("Creating Embeddings...")
            Step4_preprocess_json.run()

        st.session_state.processing = False
        st.success("Processing completed! You can ask questions now.")

# ---------------- QUESTION ANSWERING ----------------
st.header("Ask Question About Video")

# 1. Check if the file even exists
if not os.path.exists("Step5_embeddings.joblib"):
    if st.session_state.processing:
        st.warning("Video is still processing. Please wait.")
    else:
        st.info("Upload and click 'Process Video' to get started.")
else:
    # 2. Try to load the file safely
    try:
        df = joblib.load("Step5_embeddings.joblib")

        # 3. Verify the file actually has data and the 'embedding' column
        if df.empty or "embedding" not in df.columns:
            st.error(
                "The database is empty or invalid. Please click 'Process Video' again."
            )
        else:
            # 4. If everything is valid, show the question UI
            user_question = st.text_input("Ask your question:", key="user_q_input")

            if st.button("Get Answer"):
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
                        {top_chunks[["start", "end", "text"]].to_json(orient="records")}
                        ---------------------------------
                        User Question: "{user_question}"
                        Answer in a human way and guide the user to exact video timestamps.
                        """

                        # LLM Inference
                        response = inference(prompt)["response"]
                        st.markdown("### Answer")
                        st.write(response)

    except Exception as e:
        print(f"Debug Error: {e}")
        # This catches errors if the file is being written to while we try to read it
        st.info("Refreshing database... Please wait a moment.")
