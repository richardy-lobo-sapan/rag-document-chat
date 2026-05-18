import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG System",
    page_icon="📄",
    layout="centered"
)

# ── SESSION STATE ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("RAG-powered document Q&A")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        st.session_state.messages = []
        st.session_state.doc_name = uploaded_file.name

        try:
            # Import heavy libraries only when needed
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            from langchain.chains import RetrievalQA
            from langchain.embeddings.base import Embeddings
            from google import genai
            from typing import List

            class GeminiEmbeddings(Embeddings):
                def __init__(self):
                    self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                    self.model = "models/gemini-embedding-001"

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    embeddings = []
                    for text in texts:
                        result = self.client.models.embed_content(
                            model=self.model, contents=text)
                        embeddings.append(result.embeddings[0].values)
                    return embeddings

                def embed_query(self, text: str) -> List[float]:
                    result = self.client.models.embed_content(
                        model=self.model, contents=text)
                    return result.embeddings[0].values

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Reading PDF..."):
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

            with st.spinner(f"Splitting {len(pages)} pages..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(pages)

            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                embeddings = GeminiEmbeddings()
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )

            with st.spinner("Setting up QA..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                st.session_state.qa_chain = qa_chain

            st.success(f"✅ {len(pages)} pages, {len(chunks)} chunks")

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.doc_name:
        st.divider()
        st.caption(f"📄 {st.session_state.doc_name}")
        st.caption(f"💬 {len(st.session_state.messages)} messages")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + ChromaDB + Gemini")

# ── MAIN CHAT ─────────────────────────────────────────────────
st.title("📄 DocChat")
st.caption("Upload a PDF and ask questions about it")
st.divider()

if not st.session_state.qa_chain:
    st.info("👈 Upload a PDF in the sidebar to get started")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.caption(f"Chunk {i+1} — Page {source.metadata.get('page', 0) + 1}")
                        st.text(source.page_content[:200] + "...")

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources):
                            st.caption(f"Chunk {i+1} — Page {source.metadata.get('page', 0) + 1}")
                            st.text(source.page_content[:200] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {e}")