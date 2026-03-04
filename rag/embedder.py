"""
rag/embedder.py
---------------
Takes a list of LangChain Documents, embeds them using Google Gemini embeddings,
and stores them in a local ChromaDB instance, replacing any previously uploaded documents.
"""

import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import chromadb

COLLECTION_NAME = "pdf_store"

def embed_and_store(docs: list[Document]) -> Chroma:
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found. Did you set up your .env file?")

    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key
    )

    # Initialize standard ChromaDB client
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # Always delete the existing collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create the vector store from documents
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        client=chroma_client
    )

    return vectorstore


if __name__ == "__main__":
    import sys
    
    # We create some mock documents to verify the embedder independently
    print("\nInitializing Mock Documents for Embedder Verification...")
    print("──────────────────────────────────────────────────")
    
    mock_docs = [
        Document(page_content="Cats are great pets. They are very independent.", metadata={"page": 1}),
        Document(page_content="Dogs are very loyal and energetic animals.", metadata={"page": 1}),
        Document(page_content="The sky is blue because of Rayleigh scattering.", metadata={"page": 2}),
        Document(page_content="Water boils at 100 degrees Celsius at sea level.", metadata={"page": 3}),
    ]

    try:
        print("Embedding and storing documents...")
        vectorstore = embed_and_store(mock_docs)
        
        print("Testing similarity search for 'pet'...")
        results = vectorstore.similarity_search("pet", k=3)
        
        print("\n✓ Verification successful!")
        print(f"Results returned: {len(results)} (Expected 3)")
        print("\nTop 3 results:")
        for i, res in enumerate(results):
            print(f"[{i}] {res.page_content}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
