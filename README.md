# 🧠 Qwen2.5 PDF RAG Chatbot

This project is a **PDF-based Retrieval-Augmented Generation (RAG) chatbot** built on top of the **Qwen2.5B** local language model via `llama.cpp`. It allows users to upload any PDF and ask questions about its content in natural language through a **Streamlit chat UI**. The system supports **multi-turn conversations**, toggling between local and OpenAI-based LLMs, and performs efficient **semantic search** using **OpenAI embeddings** and **ChromaDB**.

---

## 🚀 Features

- 🔍 **PDF Chunking**: Smart splitting with sliding window logic
- 💡 **Embedding & Vector Storage**: Uses OpenAI `text-embedding-ada-002` + ChromaDB
- 🧠 **Local LLM Chatbot**: Qwen 2.5B (GGUF model via llama.cpp)
- 🌐 **OpenAI API Fallback**: Optional toggle to use OpenAI instead of local LLM
- 📑 **Multi-turn Context Retention**
- 🧾 **Prompt Templates** for structured response generation
- ⚡ **Streamlit UI** with chat and file upload

---

## 🏗️ Architecture

You can find the detailed system architecture and flow in [`docs/architecture.md`](docs/architecture.md).

---

## 🧰 Tech Stack

- Python 3.10+
- [Qwen 2.5B Chat GGUF](https://github.com/QwenLM/Qwen2.5) via `llama.cpp`
- OpenAI Embeddings (`text-embedding-ada-002`)
- ChromaDB (for vector search)
- PyMuPDF (`fitz`) for PDF text extraction
- Streamlit for UI

---

## ⚙️ Setup Instructions

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/yourusername/qwen2.5-pdf-rag.git
cd qwen2.5-pdf-rag
pip install -r requirements.txt
```

### 2. Download and Place Qwen2.5 GGUF Model
Place your `.gguf` file inside the project root or reference it inside `app.py`.

### 3. Add Your OpenAI Key
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## 💬 Example Use Cases

- "Summarize what this PDF is about."
- "What are the key milestones mentioned?"
- "What does the author say about climate change?"
- "List all tables from Chapter 2."

---

## 📁 Project Structure

```
qwen2.5-pdf-rag/
├── app.py                      # Main entrypoint for Streamlit app
├── modules/                   # Core logic files
│   ├── pdf_chunker.py
│   ├── embed_chunks.py
│   └── search_chunks.py
├── prompts/
│   └── final_prompt.txt       # Structured prompt format
├── docs/
│   └── architecture.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧪 Coming Soon

- Evaluation scripts for response quality
- Better caching for embeddings
- LangChain version
- CI/CD with GitHub Actions

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Credits

- [Qwen by Alibaba](https://github.com/QwenLM)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [OpenAI](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
