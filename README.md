# 📄 Qwen2.5 PDF RAG Chatbot

This is a lightweight, locally runnable **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload any PDF and ask questions about it. The system supports **multi-turn chat**, **OCR on images**, and **hybrid model toggle** between a **local Qwen2.5 GGUF model** and the **OpenAI GPT-3.5 API**.

---

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **PDF Parsing**: PyMuPDF (`fitz`) + `pytesseract` (OCR)
- **Chunking**: Fixed + Sliding Window Chunker
- **Embedding Model**: `text-embedding-ada-002` via OpenAI
- **Vector DB**: ChromaDB (Local)
- **Local LLM**: Qwen1.5–0.5B Chat (GGUF, llama.cpp)
- **Cloud LLM**: OpenAI GPT-3.5
- **Prompt Management**: Custom chain with context injection
- **Chat Memory**: JSON history per session

---

## 🧠 Architecture Overview

```mermaid
flowchart TD
    A[PDF Upload or Drop] --> B[extract_text_and_images()]
    B --> C[Chunk with Overlap]
    C --> D[embed_chunks → ada-002]
    D --> E[embedded_chunks.json → ChromaDB]
    F[User Query] --> G[get_embedding(query)]
    G --> H[ChromaDB → Top Chunk]
    H --> I[Prompt Builder + Context Injection]
    I --> J[Run Qwen Locally or GPT-3.5 API]
    J --> K[Answer Displayed on Streamlit]
