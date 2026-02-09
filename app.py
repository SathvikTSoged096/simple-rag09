import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page config
st.set_page_config(
    page_title="Simple RAG System",
    layout="centered"
)

st.title("📚 Simple RAG-based Q&A System")
st.write("Ask questions from your uploaded documents")

# Load embeddings & FAISS only once
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vectorstore()

# User input
query = st.text_input("Enter your question:")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        docs = db.similarity_search(query, k=3)

        st.subheader("🔍 Retrieved Context")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content[:500] + "...")