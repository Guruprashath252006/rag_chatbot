import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

llm = Ollama(model="llama3") if USE_OLLAMA else OpenAI(temperature=0)

st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("💬 Resume Chatbot — Chat with uploaded resumes")

uploaded_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_text = ""
        for f in uploaded_files:
            pdf = PdfReader(f)
            for page in pdf.pages:
                all_text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(all_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        retriever = vectorstore.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.chat_input("Ask about any candidate or detail...")

        if user_input:
            with st.spinner("Generating answer..."):
                result = qa({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append((user_input, result["answer"]))

        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)
