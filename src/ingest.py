import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data/knowledge_base.txt", "r") as file:

    documents = [line.strip() for line in file.readlines() if line.strip()]

print(f"Loaded {len(documents)} documents")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

document_embeddings = [get_embedding(doc) for doc in documents]
print("Embeddings created")

dimension = len(document_embeddings[0])

index = faiss.IndexFlatL2(dimension)

vectors = np.array(document_embeddings).astype("float32")
index.add(vectors)

faiss.write_index(index, "vector_store/faiss_index.index")

with open("vector_store/documents.txt", "w") as file:
    for doc in documents:
        file.write(doc + "\n")

print("FAISS index and documents saved successfully")


