from src.ingest import ensure_vector_store
from src.rag_pipeline import RAGPipeline

ensure_vector_store()
rag = RAGPipeline()

print("AI Vector Chatbot started.")
print("Type 'exit' to quit.")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    answer = rag.generate_answer(user_input)

    print("\nBot:")
    print(answer)