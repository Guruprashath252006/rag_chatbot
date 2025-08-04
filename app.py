import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Setup page
st.set_page_config(page_title="🤖 Online Chatbot Assistant", layout="wide")
st.markdown("""
    <style>
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-message {
            background-color: #F1F0F0;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Online Chatbot Assistant")
st.caption("A friendly assistant powered by LangChain and OpenAI")

# Sidebar
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.markdown("""
    <hr>
    👋 Ask me anything based on the PDF.
    """)

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to process PDF
def process_pdf(file):
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Process file and initialize chain
if uploaded_file:
    with st.spinner("Processing PDF and building vector store..."):
        db = process_pdf(uploaded_file)
        chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff")

    # Chat input and response
    query = st.chat_input("Ask something...")

    if query:
        st.session_state.history.append(("user", query))
        docs = db.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.session_state.history.append(("bot", response))

    # Display messages
    for role, message in st.session_state.history:
        if role == "user":
            st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message}</div>', unsafe_allow_html=True)

else:
    st.info("Please upload a PDF to get started.")
