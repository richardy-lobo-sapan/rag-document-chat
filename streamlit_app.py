import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DocChat - RAG System",
    page_icon="📄",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.title("📄 DocChat")
    st.caption("RAG-powered document Q&A")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        st.session_state.messages = []
        st.session_state.doc_name = uploaded_file.name

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            from langchain_core.embeddings import Embeddings
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
                st.session_state.qa_chain = {"llm": llm, "retriever": retriever}

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
                    qa = st.session_state.qa_chain
                    sources = qa["retriever"].invoke(prompt)
                    context = "\n\n".join([doc.page_content for doc in sources])
                    full_prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {prompt}"
                    response = qa["llm"].invoke(full_prompt)
                    answer = response.content

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