# SmartDocs AI: GenAI-Powered Knowledge Retrieval using RAG
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Upload and parse PDF documents
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
            continue  # Skip unsupported file types
        documents.extend(loader.load())
    return documents

# Vectorize documents
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Build QA chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Streamlit app
st.set_page_config(page_title="SmartDocs AI")
st.title("SmartDocs AI – RAG-based Knowledge Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or DOCX documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

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
    st.info("Please upload PDF files to get started.")
