import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("📄 RAG System – PDF Question Answering")

st.write(
    "Upload one or more PDF files. The system will index them and allow you to ask questions."
)

# ----------------------------
# Upload PDFs
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ----------------------------
# Create vector store
# ----------------------------
@st.cache_resource
def create_vectorstore(uploaded_files):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    db = FAISS.from_documents(chunks, embeddings)
    return db


# ----------------------------
# RAG UI
# ----------------------------
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

    db = create_vectorstore(uploaded_files)

    query = st.text_input("Ask a question from the uploaded documents:")

    if query:
        docs = db.similarity_search(query, k=3)

        st.subheader("🔍 Retrieved Answer")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc.page_content)

else:
    st.info("Please upload at least one PDF file to begin.")
