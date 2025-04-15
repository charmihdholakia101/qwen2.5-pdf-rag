import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# ✅ Load environment variables from .env
load_dotenv()
client = OpenAI()

# ✅ Load your embedded chunks
with open("embedded_chunks.json", "r") as f:
    embedded_data = json.load(f)

# ✅ Setup ChromaDB (in-memory)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.create_collection(name="pdf_chunks")

# ✅ Insert embeddings into ChromaDB
for item in embedded_data:
    collection.add(
        ids=[str(item["chunk_id"])],
        documents=[item["text"]],
        embeddings=[item["embedding"]]
    )

# ✅ Initialize chat history
chat_history = []

print("\n🤖 Chat session started! Type 'exit' to end.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Ending chat.")
        break

    # ✅ Embed the query
    query_vector = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    ).data[0].embedding

    # ✅ Retrieve relevant chunk
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    matched_chunk = results["documents"][0][0]

    # ✅ Save turn to chat history
    chat_history.append({
        "context": matched_chunk,
        "question": query
    })

    # ✅ Rebuild final prompt with chat history
    final_prompt = "You are an assistant answering questions from a PDF document.\n\n"
    for turn in chat_history:
        final_prompt += f"Context:\n{turn['context']}\n"
        final_prompt += f"Question: {turn['question']}\nAnswer:\n"

    # ✅ Save to file
    with open("final_prompt.txt", "w") as f:
        f.write(final_prompt)

    print("✅ Prompt updated in final_prompt.txt — Run Qwen to get response\n")
