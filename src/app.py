import os
from pathlib import Path
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = _PROJECT_ROOT / "vector_store"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = faiss.read_index(str(VECTOR_STORE_DIR / "faiss_index.index"))

documents = []

with open(VECTOR_STORE_DIR / "documents.txt", "r", encoding="utf-8") as file:
    content = file.read()

chunks = content.split("---END---")

for chunk in chunks:
    if chunk.strip():
        documents.append(chunk.strip())

print(f"Loaded {len(documents)} document chunks from vector store")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_context(query, top_k=3):
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        results.append(documents[i])
    return "\n".join(results)

def generate_answer(query):
    context = search_context(query)
    prompt = f"""

You are an AI assistant that answers questions using the provided context.

Answer only using the context below.

If the answer is not available in the context, say:

"I could not find that information in the provided documents."

Context:

{context}

Question:

{query}

"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful RAG-based AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

print("AI RAG Chatbot started. Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    answer = generate_answer(user_input)
    print("\nBot:", answer)