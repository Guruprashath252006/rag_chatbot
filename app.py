import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# --- Function to read PDF ---
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# --- Load & Embed PDF ---
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# --- Streamlit UI ---
st.set_page_config(page_title="Candidate Chatbot", layout="wide")
st.title("📄 RAG Chatbot for Candidate Info")

uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vector_store(text)
        retriever = vectorstore.as_retriever()
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("PDF processed successfully!")

    user_question = st.text_input("Ask a question about any candidate (e.g., 'Tell me about John Doe')")
    if user_question:
        with st.spinner("Fetching answer..."):
            response = qa_chain.run(user_question)
            st.write("### 📌 Answer:")
            st.markdown(response)
