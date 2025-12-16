import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# -----------------------------
# PAGE CONFIG (Mobile Friendly)
# -----------------------------
st.set_page_config(
    page_title="Our Chatbot",
    layout="centered"
)

st.title("📄 Our Chatbot")

# -----------------------------
# PASSWORD PROTECTION
# -----------------------------
PASSWORD = "Teju@7272"   # 🔐 change this

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("🔐 Teju@7272", type="password")
    if st.button("Login"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Upload PDF")
    file = st.file_uploader("Upload a PDF", type="pdf")
    st.info("Now you can search whatever you want")

# -----------------------------
# PROCESS PDF
# -----------------------------
if file:
    with st.spinner("Processing PDF..."):
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)

    # FREE embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Local LLM
    llm = Ollama(model="phi3")

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using ONLY the context.
If not found, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    question = st.text_input("Ask a question from the document")

    if question:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        answer = llm.invoke(
            prompt.format(context=context, question=question)
        )

        st.subheader("Answer")
        st.write(answer)


