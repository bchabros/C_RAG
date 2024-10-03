from dotenv import load_dotenv
from graph.graph_rag import crag_app, self_rag_app, adaptive_app

load_dotenv()


if __name__ == "__main__":
    options = {"1": crag_app, "2": self_rag_app, "3": adaptive_app}

    print("Choose an option:")
    print("1 - C-RAG")
    print("2 - Self-RAG")
    print("3 - Adaptive RAG")

    choice = input("Enter your choice (1-3): ")
    if choice in options:
        question = input("Enter your question: ")
        app = options[choice]
        print(app.invoke(input={"question": question}))
    else:
        print("Invalid choice. Please run the program again and select a valid option.")
