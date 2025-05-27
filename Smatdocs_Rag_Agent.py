# SmartDocs AI: GenAI-Powered Knowledge Retrieval using Local Hugging Face Pipeline

# Requirements (install before running):
# pip install langchain langchain-community streamlit python-dotenv tiktoken sentence-transformers transformers accelerate unstructured python-docx

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st

# Load environment variables (if needed for other keys)
load_dotenv()

# Initialize Local Hugging Face Pipeline
local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# Upload and parse documents
def load_documents(mixed_folder):
    documents = []
    for filename in os.listdir(mixed_folder):
        path = os.path.join(mixed_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

# Vectorize documents using Hugging Face embeddings
def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Build QA chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Streamlit app
st.set_page_config(page_title="SmartDocs AI")
st.title("SmartDocs AI – RAG-based Knowledge Assistant (Local HF Pipeline)")

uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        docs = load_documents(temp_dir)
        vectorstore = create_vectorstore(docs)
        qa_chain = build_qa_chain(vectorstore)

    st.success("Documents processed! Ask your question below.")
    user_query = st.text_input("Ask a question about the documents:")

    if user_query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": user_query})
            st.markdown("**Answer:**")
            st.write(result['result'])

            with st.expander("Sources"):
                for doc in result['source_documents']:
                    st.markdown(f"- {doc.metadata['source']}")
else:
    st.info("Please upload PDF, TXT, or DOCX files to get started.")
