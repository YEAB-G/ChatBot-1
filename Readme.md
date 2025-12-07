

````markdown
# RAG Chatbot – Chat with Your PDFs

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload your own documents and ask questions about them.

It uses:

- 🧠 **SentenceTransformers** for local text embeddings  
- 📦 A simple in-memory **vector index** (NumPy + cosine similarity)  
- 🤖 **Groq Llama 3.1** (free-tier) as the LLM  
- 🧾 Supports **PDF, TXT, and MD** files  
- 🎛 Built with **Streamlit** for a clean, modern UI  

---

## 🔍 Features

- Upload one or more **PDF / TXT / MD** files  
- Text is automatically **extracted, chunked, and embedded**  
- A small vector store lets the app **search the most relevant chunks**  
- The chatbot answers using **ONLY the retrieved context** (RAG)  
- Right-hand panel shows:
  - the **last question**
  - the **top retrieved chunks**
  - similarity scores for each chunk  

This makes it easy to explain and demo how RAG works in practice.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – UI and app framework
- **SentenceTransformers** – `all-MiniLM-L6-v2` for embeddings
- **NumPy** – vector math & cosine similarity
- **Groq** – Llama 3.1 for generation
- **PyPDF** – extract text from PDFs

---

## 🚀 Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key (free-tier)

Get a Groq API key from the Groq console, then set it:

**Windows PowerShell:**

```powershell
$env:GROQ_API_KEY="gsk-your-key-here"
```

**macOS / Linux:**

```bash
export GROQ_API_KEY="gsk-your-key-here"
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

Then open the URL that Streamlit prints (usually `http://localhost:8501`).

---

## 🧪 How to Use

1. Upload one or more **PDF / TXT / MD** documents from the sidebar
2. Click **“Build / Rebuild Vector Index”**
3. Ask questions in the chat box, e.g.:

   * “Summarize this document.”
   * “What are the main points in this chapter?”
   * “What does it say about X?”
4. Check the **“Retrieval & Context”** panel to see which chunks were used.

This is a good example of a **practical RAG pipeline** you can show in a portfolio or during interviews.

---

## 💼 Why this is interesting for a portfolio

* Shows you understand **RAG** (not just a plain chatbot)
* Uses a **modern open LLM** (Llama 3.1 via Groq)
* Demonstrates:

  * document ingestion (PDF, TXT, MD)
  * chunking
  * embeddings
  * vector search
  * prompt construction
  * UI for both chat and retrieved context

````

Just replace `<your-username>` and `<your-repo-name>` with your actual ones.

Then:

```powershell
git add README.md
git commit -m "Add project README"
git push
````




