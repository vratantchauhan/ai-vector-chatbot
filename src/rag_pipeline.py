import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from src.config.settings import INDEX_PATH, DOCS_PATH, OPENAI_API_KEY

load_dotenv()

client = OpenAI(api_key=OPENAI_API_KEY)


class RAGPipeline:

    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.documents = self.load_documents()

    def load_documents(self):
        documents = []

        with open(DOCS_PATH, "r", encoding="utf-8") as file:
            content = file.read()

        chunks = content.split("---END---")

        for chunk in chunks:
            if chunk.strip():
                documents.append(chunk.strip())

        return documents

    def get_embedding(self, text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding

    def search_context(self, query, top_k=3):

        query_embedding = np.array(
            [self.get_embedding(query)]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for i in indices[0]:
            results.append(self.documents[i])

        return "\n\n".join(results)

    def generate_answer(self, query):

        context = self.search_context(query)

        prompt = f"""
You are an AI assistant that answers questions using the provided context.

Answer ONLY using the context below.

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
                {
                    "role": "system",
                    "content": "You are a helpful RAG-based AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content