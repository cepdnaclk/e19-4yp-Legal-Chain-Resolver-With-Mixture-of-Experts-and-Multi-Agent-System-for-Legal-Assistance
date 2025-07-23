# --- Imports and Setup ---
import os
import re
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Load environment variables ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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

# --- LLM Setup (Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer. 
Explain the answer in a way that is easy to understand using only the provided context.
I will tip you $1000 if the user finds the answer helpful. 
Do NOT include phrases like "Based on the context" or "Here's the answer".
ONLY return the answer.
If context is not provided or empty, respond with "Please ask a question".

<context>
{context}
</context>

Question: {input}
""")

# --- Document Chain and Retriever ---
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 6})
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# --- Interactive User Input Loop ---
if __name__ == "__main__":
    print("Ask a question about the documents. Type 'exit' or 'quit' to stop.")
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break
        try:
            # Show retrieved docs (optional, for inspection)
            retrieved_docs = retriever.invoke(user_query)
            i = 0
            for doc in retrieved_docs:
                i += 1
                print(f"\nRetrieved Document[{i}]: {doc.page_content[:500]}...")  # show first 500 chars

            # Run through LLM chain
            response = retrieval_chain.invoke({"input": user_query})
            print("\nAnswer:", response['answer'])
        except Exception as e:
            print(f"An error occurred: {e}")