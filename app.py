import os
import hashlib
import streamlit as st
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from google.colab import userdata
import os

os.environ["MISTRALAI_API_KEY"] = "YOUR_API_KEY"

# LlamaIndex Settings
Settings.llm = MistralAI(api_key="YOUR_API_KEY", temperature=0.1, model='open-mixtral-8x22b')
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=256)
Settings.num_output = 512
Settings.context_window = 3900


# Ensure the storage directory exists
if not os.path.exists('./storage'):
    os.makedirs('./storage')

# Function to load documents from a directory and add user metadata
def load_documents(directory, user):
    reader = SimpleDirectoryReader(input_dir=directory)
    documents = reader.load_data()
    for doc in documents:
        doc.metadata["user"] = user
    return documents

# Function to create or load an index for a user
def create_or_load_index(user: str, documents=None):
    """Create a new index or load an existing one for a user."""
    storage_dir = f"./storage/{user}"
    if os.path.exists(storage_dir):
        # Load index from storage if it already exists
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
    else:
        if documents is None:
            raise ValueError("Documents must be provided to create a new index.")
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=256)]
        )
        nodes = pipeline.run(documents=documents)
        index = VectorStoreIndex(nodes)
        # Persist the index
        index.storage_context.persist(persist_dir=storage_dir)

    return index

# Create query engine for a specific user with strict filtering
def create_query_engine(index, user):
    retriever = index.as_retriever(
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="user", value=user)]
        ),
        similarity_top_k=3
    )
    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.45)],
        response_mode="tree_summarize",
        use_strict_synthesis=True,  # Prevent answering outside document scope
        fallback_message="No relevant information found in the document."
    )

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Ensure session state is properly initialized
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None  # Keeps track of logged-in user

# Streamlit Interface
st.title("Multi-Tenant Document Query System")

# Main function for user interaction
def main():
    if st.session_state.logged_in_user is None:
        st.sidebar.title("User Authentication")

        # User login or registration
        user_action = st.sidebar.selectbox("Select Action", ["Login", "Register"])

        if user_action == "Register":
            new_user = st.sidebar.text_input("Enter your name")
            new_password = st.sidebar.text_input("Create Password", type="password")
            doc_directory = st.sidebar.text_input("Enter Document Directory Path")

            if st.sidebar.button("Register"):
                if new_user and new_password and doc_directory:
                    hashed_password = hash_password(new_password)

                    # Load or create index
                    try:
                        documents = load_documents(doc_directory, new_user)
                        index = create_or_load_index(new_user, documents)
                        st.session_state.user_data[new_user] = {
                            "password": hashed_password,
                            "engine": create_query_engine(index, new_user)
                        }
                        st.sidebar.success(f"User {new_user} registered successfully!")
                    except ValueError as e:
                        st.sidebar.error(str(e))
                else:
                    st.sidebar.error("Please provide all fields.")

        elif user_action == "Login":
            existing_user = st.sidebar.text_input("Username")
            existing_password = st.sidebar.text_input("Password", type="password")

            if st.sidebar.button("Login"):
                if existing_user in st.session_state.user_data:
                    hashed_password = hash_password(existing_password)
                    if st.session_state.user_data[existing_user]["password"] == hashed_password:

                        # Load the user's index
                        index = create_or_load_index(existing_user)

                        # Set the user as logged in and load their query engine
                        st.session_state.logged_in_user = existing_user
                        st.session_state.user_data[existing_user]["engine"] = create_query_engine(index, existing_user)
                        st.sidebar.success(f"Welcome back, {existing_user}!")
                    else:
                        st.sidebar.error("Incorrect password.")
                else:
                    st.sidebar.error(f"User {existing_user} does not exist. Please register.")
    else:
        # Display query input for logged-in user
        user_query(st.session_state.logged_in_user)
        if st.button("Logout"):
          st.session_state.logged_in_user = None  # Log out the user
          st.query_params.clear()  # Clear the URL query parameters to refresh the page



# Query function for logged-in users
def user_query(user):
    st.subheader(f"Query for {user}")
    query = st.text_input("Enter your query", "")

    if st.button("Submit Query"):
        if query:
            # Perform the query and capture the response object
            response = st.session_state.user_data[user]["engine"].query(query)

            # Let's inspect the response object to understand its structure
            try:
                if hasattr(response, 'response') and response.response.lower() in ["empty response", "no response", ""]:
                    st.warning("No relevant information was found in the document.")
                else:
                    st.success(f"Response: {getattr(response, 'response', 'No response available')}")

                # Check for source nodes and extract file names
                if not hasattr(response, 'source_nodes') or not response.source_nodes:
                    st.info("No source documents found.")
                else:
                    source_info = [
                        f"{node.node.metadata.get('file_name', 'Unknown file')} (page {node.node.metadata.get('page_label', 'Unknown page')})"
                        for node in response.source_nodes
                    ]
                    st.info(f"Source(s): {', '.join(source_info)}")
            except AttributeError as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()