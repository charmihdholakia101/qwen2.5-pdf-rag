---
marp: true
---

# 🧠 Qwen2.5 RAG Chatbot Architecture

---

## 📦 Phase 1: Ingestion and Embedding

```mermaid
flowchart TD
    A[User uploads PDF or drops it into app] --> B[extract_text_and_images() from app.py/pdf_chunker.py]
    B --> C[chunk_text() with overlap]
    C --> D[embed_chunks.py → get_embedding() using ada-002]
    D --> E[embedded_chunks.json saved]
    E --> F[search_chunks.py loads this JSON]
```

---

## 📂 Phase 2: Vector Storage and Search

```mermaid
flowchart TD
    A[embedded_chunks.json] --> B[search_chunks.py or app.py → load chunks]
    B --> C[ChromaDB: create collection + insert embeddings]
    C --> D[User types a question (Streamlit input)]
    D --> E[get_embedding() of query]
    E --> F[ChromaDB query: top 1 result]
```

---

## 🔮 Phase 3: Prompt + Generation

```mermaid
flowchart TD
    A[Matched chunk + query] --> B[Prompt builder → build_prompt()]
    B --> C[final_prompt.txt saved]
    C --> D[Local Qwen call via llama-cli OR GPT-3.5 API]
    D --> E[Output displayed on Streamlit UI]
