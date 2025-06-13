import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, Runnable
import tempfile
import os

# 🔐 Hardcoded API Key for testing
GOOGLE_API_KEY = "AIzaSyBYTDmlXHHg9SGYDBA-mimwffrYuSruLZc"

# Initialize Gemini model (2.0 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Streamlit UI
st.set_page_config(page_title="Gemini RAG QA App", page_icon="📄")
st.title("📄🔍 Gemini RAG Question Answering App")
st.markdown("Upload a PDF and ask questions from it using **RAG + Gemini**.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        with st.spinner("📚 Processing PDF..."):
            # Load and split document
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(pages)

            # Create embeddings and FAISS vector store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            retriever = vectorstore.as_retriever()

            # Final RAG chain
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])

            rag_chain: Runnable = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
            )

            if query:
                with st.spinner("🧠 Thinking..."):
                    result = rag_chain.invoke(query)
                    st.success("✅ Answer:")
                    st.write(result.content)

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)

    finally:
        os.remove(pdf_path)
