import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Simple RAG App")
st.title("📘 Simple RAG System")

PDF_PATH = "data/notes.pdf"
INDEX_PATH = "faiss_index"

@st.cache_resource
def load_or_create_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(INDEX_PATH):
        st.info("Creating vector index for the first time...")

        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_PATH)
    else:
        db = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return db

db = load_or_create_db()

query = st.text_input("Ask a question from the notes:")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrieved Answer")
    for doc in docs:
        st.write(doc.page_content)
