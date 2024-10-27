### **README.md**

# Multi-Tenant Healthcare RAG Chatbot

This repository contains a multi-tenant Retrieval-Augmented Generation (RAG) chatbot specifically designed for the healthcare domain. It enables authenticated medical professionals to retrieve relevant information from specialty-specific healthcare documents, such as cardiology and neurology records, facilitating informed decision-making in patient care.

## Key Features

- **Multi-Tenant Security**: Ensures users can access only their own authorized documents, with strict metadata filtering and user-based indexing in a single vector store.
- **RAG Pipeline**: Utilizes Mistral AI's open-mixtral-8x22b model and `BAAI/bge-base-en-v1.5` embeddings for contextually accurate responses.
- **Scalable Ingestion Pipeline**: Document chunking and indexing enable high scalability for new document collections.
- **User Authentication & Interface**: Built using Streamlit, providing a simple login interface with query and response capabilities.

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- API key for Mistral AI (create an account at [Mistral AI](https://mistral.ai/))
  
### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/SohamNarsale/multi-tenant-healthcare-rag-chatbot.git
    cd multi-tenant-healthcare-rag-chatbot
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up your environment variable for Mistral AI API key:
    ```bash
    export MISTRALAI_API_KEY='your_api_key_here'
    ```

### Running the Application

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
2. Open the Streamlit application in your browser, register/login, and start querying your documents.

## Usage

1. **Registration**: Enter your username, password, and document directory path to register.
2. **Login**: Existing users can log in using their credentials.
3. **Querying**: After login, enter your query to retrieve relevant healthcare information from your assigned documents. 

### Technical Details

- **Document Ingestion**: Documents are loaded and assigned metadata for multi-tenant filtering.
- **Indexing and Retrieval**: `VectorStoreIndex` is built per user, with filters to ensure users can only access their assigned documents.
- **Query Processing**: User queries are processed via `RetrieverQueryEngine`, which filters based on metadata and restricts answers to indexed content.

### Sample Code

A simplified version of the code can be found in the `app.py` file, covering the ingestion, indexing, and query functionality.

---

### **requirements.txt**

```plaintext
streamlit==1.20.0
llama-index==0.8.3
llama-index-llms-mistralai==0.3.5
llama-index-embeddings-huggingface==0.3.5
pypdf==3.5.2
hashlib
```