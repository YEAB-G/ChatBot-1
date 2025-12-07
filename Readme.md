# RAG Chatbot (Groq + Local Embeddings + PDF Support)

- Upload `.txt`, `.md`, or `.pdf` documents.
- They are split into chunks and embedded locally (SentenceTransformers).
- A simple in-memory vector index is used to search relevant chunks.
- Groq Llama 3.1 (free tier) answers using **only** the retrieved context.

## Setup

```bash
pip install -r requirements.txt
