from src.rag_pipeline import RAGPipeline

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