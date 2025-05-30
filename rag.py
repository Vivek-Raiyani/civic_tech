import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st

# load_dotenv()
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
# 📍 Setup Gemini Embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
)

# 🧠 Text Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

PDF_FOLDER = "pdfs/"
documents = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()

        # Add metadata to each page
        for i, doc in enumerate(pages):
            doc.metadata.update({
                "source": filename,
                "page_number": doc.metadata.get("page", i + 1),
                "document_type": "safety_faq",
                "agency": "FEMA",  # customize if needed
                "category": "disaster_preparedness"
            })

        chunks = text_splitter.split_documents(pages)
        documents.extend(chunks)

# 🧠 Store into ChromaDB
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="chroma_store_gemini"
)

vectordb.persist()
print("✅ Data embedded with Gemini and stored in ChromaDB.")
