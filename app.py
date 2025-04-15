import streamlit as st
import os
import io
import subprocess
import json
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# Load environment variables
load_dotenv()
client = OpenAI()

# Persistent ChromaDB setup
chroma_client = PersistentClient(path=".chromadb")
collection = chroma_client.get_or_create_collection("chat_pdf")

# --- PDF Parsing & OCR ---
def extract_text_and_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_text = ""
    image_blocks = []

    for page_num, page in enumerate(doc):
        all_text += page.get_text()

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ocr_text = pytesseract.image_to_string(image)

            image_blocks.append({
                "page": page_num + 1,
                "index": img_index,
                "ocr_text": ocr_text.strip(),
                "image_obj": image
            })

    return all_text, image_blocks

def chunk_text(text, max_tokens=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_tokens
        chunks.append(text[start:end])
        start += max_tokens - overlap
    return chunks

# --- Embeddings & Storage ---
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def embed_chunks(text_chunks, image_ocr_chunks):
    try:
        existing = collection.get()
        if existing and "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as e:
        print("Warning: Could not clear collection -", str(e))

    all_chunks = text_chunks + image_ocr_chunks

    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[f"{chunk['type']}_{i}"],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"type": chunk["type"], "page": chunk.get("page", -1), "index": chunk.get("index", 0)}]
        )

    return all_chunks

def search_top_chunk(query, route_to_images=False):
    query_vector = get_embedding(query)

    if route_to_images:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            where={"type": "image"}
        )
    else:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            where={"type": "text"}
        )

    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]
    return top_doc, top_meta

# --- Prompt & Response ---
def build_prompt(chat_history):
    prompt = (
        "You are an assistant answering questions from a PDF document. "
        "If the context comes from an image, explain what the image visually conveys using the OCR text. "
        "Do not say you cannot show the image — the user is already viewing it.\n\n"
    )

    for turn in chat_history:
        meta = turn.get("meta", {})
        chunk_type = meta.get("type", "text")
        chunk_text = turn["chunk"].strip()

        if chunk_type == "image":
            prompt += f"[Image from Page {meta.get('page', '?')}]\n"
            chunk_text += "\n[This is OCR text extracted from an image that is currently visible to the user. Do not say the image is unavailable. Explain what the image conveys visually.]"

        prompt += f"Context:\n{chunk_text}\n"
        prompt += f"Question: {turn['query']}\nAnswer:\n"

    return prompt

def run_qwen_locally(prompt):
    try:
        with open("final_prompt.txt", "w") as f:
            f.write(prompt)

        result = subprocess.run(
            ["./bin/llama-cli", "-m", "../../models/qwen2.5.gguf", "-f", "../../final_prompt.txt"],
            cwd="llama.cpp/build",
            capture_output=True,
            text=True,
            timeout=90
        )
        return parse_qwen_output(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return "⚠️ Qwen timed out while generating a response."
    except Exception as e:
        return f"❌ Qwen generation failed: {str(e)}"

def run_openai_completion(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"

def parse_qwen_output(output):
    parts = output.split("Answer:")
    return parts[-1].strip() if len(parts) > 1 else "⚠️ Qwen did not generate a proper answer."

# --- Helper ---
def is_image_query(query):
    keywords = ["show", "image", "see", "diagram", "picture", "figure", "illustration", "visual"]
    return any(k in query.lower() for k in keywords)

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 PDF Chat with OCR", layout="wide")
st.title("📄 Chat with PDF + Images + OCR")

model_choice = st.radio("Select model for answers:", ["Qwen (local)", "OpenAI (cloud)"], horizontal=True)
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with st.spinner("🔍 Extracting text and images..."):
        text, image_blocks = extract_text_and_images(pdf_file)

        image_map = {(img["page"], img["index"]): img["image_obj"] for img in image_blocks}

        text_chunks = [{"type": "text", "text": chunk} for chunk in chunk_text(text)]
        image_ocr_chunks = [
            {"type": "image", "text": img["ocr_text"], "page": img["page"], "index": img["index"]}
            for img in image_blocks if img["ocr_text"]
        ]

        embed_chunks(text_chunks, image_ocr_chunks)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something about the document or its images:")

    if user_input:
        route_images = is_image_query(user_input)
        top_chunk, top_meta = search_top_chunk(user_input, route_to_images=route_images)

        st.session_state.chat_history.append({
            "query": user_input,
            "chunk": top_chunk,
            "meta": top_meta
        })

        prompt = build_prompt(st.session_state.chat_history)

        if model_choice == "Qwen (local)":
            answer = run_qwen_locally(prompt)
        else:
            answer = run_openai_completion(prompt)

        st.session_state.chat_history[-1]["response"] = answer

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            if turn["meta"]["type"] == "image":
                key = (turn["meta"]["page"], turn["meta"].get("index", 0))
                if key in image_map:
                    st.image(image_map[key], caption=f"Image from Page {turn['meta']['page']}")
            st.markdown(turn.get("response", "_Waiting for response..._"))
