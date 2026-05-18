import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from google import genai
from typing import List

load_dotenv()

# ── CUSTOM EMBEDDINGS USING NEW SDK ──────────────────────────
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "models/gemini-embedding-001"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return result.embeddings[0].values

# ── 1. LOAD PDF ──────────────────────────────────────────────
def load_pdf(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages

# ── 2. SPLIT INTO CHUNKS ─────────────────────────────────────
def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# ── 3. CREATE VECTOR STORE ───────────────────────────────────
def create_vectorstore(chunks):
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = GeminiEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created!")
    return vectorstore

# ── 4. CREATE QA CHAIN ───────────────────────────────────────
def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ── 5. ASK A QUESTION ────────────────────────────────────────
def ask(qa_chain, question):
    print(f"\nQuestion: {question}")
    result = qa_chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")
    print(f"\nSources: {len(result['source_documents'])} chunks used")
    return result

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        print("Step 1: Loading PDF...")
        pages = load_pdf(pdf_path)

        print("Step 2: Splitting into chunks...")
        chunks = split_documents(pages)

        print("Step 3: Creating vector store...")
        vectorstore = create_vectorstore(chunks)

        print("Step 4: Creating QA chain...")
        qa_chain = create_qa_chain(vectorstore)

        print("\nRAG system ready! Type 'quit' to exit.\n")
        while True:
            question = input("You: ")
            if question.lower() == "quit":
                break
            ask(qa_chain, question)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()