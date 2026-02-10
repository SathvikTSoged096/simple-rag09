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
    "Upload one or more PDF files. The system will answer questions ONLY if they are relevant to the document."
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
@st.cache_resource(show_spinner="Indexing documents...")
def create_vectorstore(uploaded_files):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    all_docs = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if docs:
            all_docs.extend(docs)

    if not all_docs:
        st.error("❌ No text could be extracted from the uploaded PDFs.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        st.error("❌ Text splitting produced no chunks.")
        st.stop()

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
        # 🔑 USE SCORED SEARCH
        results = db.similarity_search_with_score(query, k=5)

        THRESHOLD = 0.6  # lower = stricter (recommended: 0.4–0.6)

        relevant_docs = [
            doc for doc, score in results if score < THRESHOLD
        ]

        if not relevant_docs:
            st.warning("❌ This question is NOT related to the uploaded document.")
            st.stop()

        st.subheader("🔍 Retrieved Answer")
        for i, doc in enumerate(relevant_docs, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc.page_content)

else:
    st.info("Please upload at least one PDF file to begin.")
