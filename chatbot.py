import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data.txt", "r") as file:
	documents = [line.strip() for line in file.readlines() if line.strip()]
	print(documents)

def get_embedding(text):
	response = client.embeddings.create(
		model="text-embedding-3-small",
		input=text
	)
	return response.data[0].embedding

document_embeddings = [get_embedding(doc) for doc in documents]

dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)

vectors = np.array(document_embeddings).astype("float32")
index.add(vectors)

def search_context(query, top_k=2):
	query_embedding = np.array([get_embedding(query)]).astype("float32")
	distances, indices = index.search(query_embedding, top_k)

	results = []
	for i in indices[0]:
		results.append(documents[i])
	return "\n".join(results)

def generate_answer(query):
	context = search_context(query)
	prompt = f"""

You are an AI assistant. Answer the user using the context below only nothing from outside the context.

Context:

{context}

User Question:

{query}

"""
	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[
			{"role": "system", "content": "You are a helpful AI chatbot who answers from the context provided only"},
			{"role": "user", "content": prompt}
		]
	)

	return response.choices[0].message.content

print("AI Vector Chatbot started. Type 'exit' to quit.")

while True:
	user_input = input("\nYou: ")
	if user_input.lower() == "exit":
		print("Bot: Goodbye!")
		break
	answer = generate_answer(user_input)
	print("\nBot:", answer)
