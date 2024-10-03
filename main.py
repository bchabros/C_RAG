import streamlit as st
from dotenv import load_dotenv
from graph.graph_rag import crag_app, self_rag_app, adaptive_app
from PIL import Image  # Import the image library to display PNGs

load_dotenv()

# Define the options for model selection with corresponding image paths
options = {
    "C-RAG": {"app": crag_app, "image": "./png/graph_1.png"},
    "Self-RAG": {"app": self_rag_app, "image": "./png/graph_2.png"},
    "Adaptive RAG": {"app": adaptive_app, "image": "./png/graph_3.png"}
}

# Set the title of the Streamlit app
st.title("RAG Model Selector")

# Create a select box for model selection
model_choice = st.selectbox("Choose a RAG model:", list(options.keys()))

# Display the corresponding image when a model is selected
if model_choice:
    st.image(Image.open(options[model_choice]["image"]), caption=f"{model_choice} Model Graph", use_column_width=True)

# Create a text area for the user to input their question
question = st.text_area("Enter your question:")

# Create a submit button
if st.button("Submit"):
    if model_choice in options:
        # Invoke the selected RAG model
        app = options[model_choice]["app"]
        response = app.invoke(input={"question": question})
        # Display the response
        st.write("**Response:**")
        st.write(response["generation"])
    else:
        st.error("Invalid choice. Please select a valid RAG model.")
