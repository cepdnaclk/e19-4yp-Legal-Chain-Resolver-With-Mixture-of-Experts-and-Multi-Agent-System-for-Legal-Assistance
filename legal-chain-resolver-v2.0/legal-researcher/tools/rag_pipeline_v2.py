# --- Imports and Setup ---
import os
from dotenv import load_dotenv
from datetime import datetime

# Import necessary libraries from LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

# --- Vector Embedding and Vector Store (FAISS) ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
script_dir = os.path.dirname(__file__)
FAISS_INDEX_PATH = os.path.join(script_dir, "faiss_index")
data_dir = os.path.abspath(os.path.join(script_dir, "../../../data/law_documents/"))

def create_and_save_faiss_index():
    print("Creating new FAISS index...")
    # --- Data Ingestion ---
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    all_combined_docs = []
    for pdf_file in pdf_files:
        pdf_loader = PyPDFLoader(pdf_file)
        pages = pdf_loader.load()
        
        # Combine all pages of a single PDF into one document
        combined_text = "\n".join([p.page_content for p in pages])
        # Remove unnecessary whitespace
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        combined_doc = Document(page_content=combined_text, metadata={"source": pdf_file})
        all_combined_docs.append(combined_doc)

    # --- Chunking Data ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
    )
    chunked_docs = text_splitter.split_documents(all_combined_docs)
    print(f"Loaded {len(all_combined_docs)} combined documents and split into {len(chunked_docs)} chunks.")
    faiss_vector_store = FAISS.from_documents(chunked_docs, embeddings)
    faiss_vector_store.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved.")
    return faiss_vector_store

if os.path.exists(FAISS_INDEX_PATH):
    # Check if any PDF file is newer than the FAISS index
    rebuild_index = False
    faiss_index_mtime = os.path.getmtime(FAISS_INDEX_PATH)
    
    for pdf_file in [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]:
        if os.path.getmtime(pdf_file) > faiss_index_mtime:
            rebuild_index = True
            break

    if rebuild_index:
        print("PDF files have been updated. Rebuilding FAISS index...")
        faiss_vector_store = create_and_save_faiss_index()
    else:
        print("Loading existing FAISS index...")
        faiss_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
else:
    faiss_vector_store = create_and_save_faiss_index()

def retrieve_documents(query, k=6):
    retriever = faiss_vector_store.as_retriever(search_kwargs={'k': k})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

# --- Interactive User Input Loop ---
print("Ask a question about the documents. Type 'exit' or 'quit' to stop.")
while True:
    user_query = input("\nYour question: ").strip()
    if user_query.lower() in {"exit", "quit"}:
        print("Exiting. Goodbye!")
        break
    try:
        retrieved_docs = retrieve_documents(user_query)
        i = 0
        for doc in retrieved_docs:
            i += 1
            print(f"\nRetrieved Document[{i}]: {doc.page_content}")
    except Exception as e:
        print(f"An error occurred: {e}")
