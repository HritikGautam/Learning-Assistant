#  Learning Assistant - Video Q&A System

An AI-powered **Learning Assistant** that allows users to ask questions about a YouTube course and receive answers with **exact video references and timestamps**.

The system converts video subtitles into embeddings and uses **vector similarity search + LLM reasoning** to generate contextual answers.
Link:- https://ask-video-ai.streamlit.app/
---

#  Features

- Ask questions about a complete video course
- Retrieves the most relevant subtitle segments
- Uses **semantic search with embeddings**
- Generates answers using **LLM (Llama 3 via Groq)**
- Provides **video numbers and timestamps**
- Fast and interactive **Streamlit interface**

---

#  How It Works

1. Video subtitles are extracted and processed.
2. Each subtitle chunk is converted into an **embedding vector**.
3. The embeddings are stored in `embeddings.joblib`.
4. When a user asks a question:
   - The question is converted into an embedding
   - **Cosine similarity** finds the most relevant chunks
5. Relevant context is sent to the **LLM (Groq - Llama 3.3 70B)**.
6. The AI generates a contextual answer with timestamps.

---

#  Project Structure

```
Learning-Assistant
│
├── app.py                 # Main Streamlit application
├── Step1.py               # Data preprocessing step
├── Step2.py               # Subtitle extraction
├── Step3.py               # Text chunking
├── Step4.py               # Embedding generation
├── Step5.py               # Final embedding storage
│
├── embeddings.joblib      # Stored embedding vectors
│
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages for deployment
└── .gitignore
```

---

#  Tech Stack

**Frontend**
- Streamlit

**Backend / AI**
- Python
- HuggingFace Embeddings
- Sentence Transformers
- Groq LLM (Llama-3.3-70B)

**ML / Data Processing**
- NumPy
- Pandas
- Scikit-learn

**Vector Similarity**
- Cosine Similarity

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/HritikGautam/Learning-Assistant.git
cd Learning-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Setup API Key

Create a **Streamlit secrets file**.

```
.streamlit/secrets.toml
```

Add your **Groq API key**:

```toml
GROQ_API_KEY="your_api_key_here"
```

---

#  Run the Application

```bash
streamlit run app.py
```

The app will open in your browser.

---

#  Example Usage

1. Open the Streamlit interface.
2. Enter a question about the course.
3. The system will:
   - Find the most relevant video segments
   - Generate an answer
   - Provide **video number and timestamps**.

Example question:

```
What is Python Virtual Environment?
```

Example response:

```
You can learn about Python Virtual Environment in Video number 12 around timestamp 3:45.
```

---

#  Course Used

The assistant answers questions about this YouTube course:

Python Full Course Playlist  
https://youtube.com/playlist?list=PLu0W_9lII9agq5TrH9XLIKQvv0iaF2X3w

---

#  Future Improvements

- Currently it answers for a few videos, as the course is too big and creating embeddings will take time. 
- Add support for **multiple courses**
- Implement **vector databases (FAISS / Chroma)**
- Improve UI with chat interface
- Add **streaming responses**
- Deploy as a **public AI learning assistant**

---

#  Author

**Hritik Gautam**

GitHub:  
https://github.com/HritikGautam

---

⭐ If you found this project helpful, consider giving it a **star**!
