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
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mkv", "avi"],
    disabled=st.session_state.processing,
    on_change=cleanup_workspace,
)

if uploaded_file:
    # 1. Clear everything whenever a NEW file object is detected
    # We use the file ID or name to detect a change
    if st.session_state.get("last_uploaded_id") != uploaded_file.name:
        cleanup_workspace()
        st.session_state.last_uploaded_id = uploaded_file.name

    video_path = os.path.join("videos", uploaded_file.name)

    # 2. ALWAYS write the file fresh
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # use getbuffer() for better memory handling

    st.session_state.video_uploaded = True
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

# Check if the database for the CURRENT video exists
has_embeddings = os.path.exists("Step5_embeddings.joblib")

can_answer = (
    st.session_state.video_uploaded
    and not st.session_state.processing
    and has_embeddings
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

elif not has_embeddings:
    st.warning("Please process the video first.")


else:
    # Load fresh embeddings
    df = joblib.load("Step5_embeddings.joblib")

    if "embedding" not in df.columns:
        st.error("Error: Embedding column missing in data. Please re-process.")

    else:
        user_question = st.text_input("Ask your question:", key="user_q_input")

        if st.button("Get Answer"):
            if not user_question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("Thinking..."):
                    # 1. Embed query
                    question_embedding = create_embedding([user_question])[0]
                    # 2. Vector Search
                    similarities = cosine_similarity(
                        np.vstack(df["embedding"]), [question_embedding]
                    ).flatten()
                    top_k = 3
                    top_indices = similarities.argsort()[::-1][:top_k]
                    top_chunks = df.loc[top_indices]
                    # 3. Build Prompt
                    prompt = f""" Here are video subtitle chunks:

                    {top_chunks[["start", "end", "text"]].to_json(orient="records")}

                    ---------------------------------
                    User Question:
                    "{user_question}"

                    Answer in a human way and guide the user to exact video timestamps.
                    """
                    # 4. LLM Inference
                    response = inference(prompt)["response"]
                st.markdown("### Answer")
                st.write(response)
