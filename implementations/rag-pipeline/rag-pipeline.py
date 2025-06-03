# rag-pipeline-test-interactive.py

# --- Imports and Setup ---
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Import necessary libraries from LangChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Data Ingestion ---
pdf_loader = PyPDFLoader("detailed.pdf")
pdf = pdf_loader.load()

# --- Chunking Data ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)
chunked_pdf = text_splitter.split_documents(pdf)

# --- Vector Embedding and Vector Store (FAISS) ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_vector_store = FAISS.from_documents(chunked_pdf, embeddings)

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
retriever = faiss_vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# --- Interactive User Input Loop ---
print("Ask a question about the documents. Type 'exit' or 'quit' to stop.")
while True:
    user_query = input("\nYour question: ").strip()
    if user_query.lower() in {"exit", "quit"}:
        print("Exiting. Goodbye!")
        break
    try:
        response = retrieval_chain.invoke({"input": user_query})
        print("\nAnswer:", response['answer'])
    except Exception as e:
        print(f"An error occurred: {e}")
