import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize OpenAI client
client = OpenAI()

# ✅ Extract text and images from PDF
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    image_captions = []

    for page_num, page in enumerate(doc):
        full_text += page.get_text()
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            image_captions.append(f"[Image on Page {page_num + 1} - placeholder caption]")
    
    return full_text, image_captions

# ✅ Chunk text with overlap
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ✅ Call OpenAI Embedding API
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# ✅ Generate embeddings for all chunks
def generate_embeddings(all_texts):
    embedded_data = []
    for i, text in enumerate(all_texts):
        print(f"🔍 Embedding chunk {i + 1}/{len(all_texts)}")
        embedding = get_embedding(text)
        embedded_data.append({
            "chunk_id": i,
            "text": text,
            "embedding": embedding
        })
    return embedded_data

if __name__ == "__main__":
    print("🚀 Starting embedding pipeline...")

    pdf_path = "moon.pdf"  # or switch to your other PDF
    full_text, image_captions = extract_text_and_images(pdf_path)

    print(f"📄 Raw text length: {len(full_text)}")
    print(f"🖼️ Images detected: {len(image_captions)}")

    text_chunks = chunk_text(full_text)
    all_chunks = text_chunks + image_captions

    print(f"🧩 Total chunks (text + images): {len(all_chunks)}")

    embedded_chunks = generate_embeddings(all_chunks)

    with open("embedded_chunks.json", "w") as f:
        json.dump(embedded_chunks, f)

    print("✅ Embeddings generated and saved to embedded_chunks.json")
