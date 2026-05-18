# DocChat — RAG Document Q&A System

A **Retrieval Augmented Generation (RAG)** system that lets you upload any PDF and ask questions about it. Built with LangChain, ChromaDB, and Google Gemini.

🚀 **Live Demo:** https://rag-document-chat-o4uujl2eatcayam4vxmuh3.streamlit.app

---

## What It Does

Upload any PDF → ask questions in plain English → get accurate answers with source citations from the document.

User uploads PDF
↓
LangChain splits into chunks
↓
Gemini converts chunks to embeddings (vectors)
↓
ChromaDB stores vectors locally
↓
User asks a question
↓
ChromaDB finds 3 most relevant chunks
↓
Gemini answers using only those chunks
↓
Answer + source page numbers returned

---

## Why RAG?

| Pure LLM | RAG System |
|----------|-----------|
| Answers from training data | Answers from YOUR document |
| May hallucinate | Grounded in real content |
| No source citations | Shows exact chunks used |
| Can't read new documents | Works with any PDF |

---

## Tech Stack

- **LangChain** — Document loading, chunking, retrieval pipeline
- **ChromaDB** — Local vector database for storing embeddings
- **Google Gemini** — LLM for answering + embedding model
- **Streamlit** — Web UI and deployment
- **PyPDF** — PDF text extraction
- **python-dotenv** — Environment variable management

---

## How It Works

**Step 1 — Document Processing**
```python
# Load PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()  # 15 pages

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)  # 52 chunks
```

**Step 2 — Embedding & Storage**
```python
# Convert chunks to vectors
embeddings = GeminiEmbeddings()  # models/gemini-embedding-001
vectorstore = Chroma.from_documents(chunks, embeddings)
# 52 vectors stored in ChromaDB
```

**Step 3 — Retrieval & Generation**
```python
# Find relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
sources = retriever.invoke(question)

# Generate answer
context = "\n\n".join([doc.page_content for doc in sources])
answer = llm.invoke(f"Answer using context:\n{context}\n\nQuestion: {question}")
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/richardy-lobo-sapan/rag-document-chat.git
cd rag-document-chat

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run
streamlit run streamlit_app.py
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

---

## Project Structure

rag-document-chat/
├── rag.py              # Terminal version (core logic)
├── streamlit_app.py    # Browser UI with file upload
├── requirements.txt    # Dependencies
├── .env                # API key (not on GitHub)
└── .gitignore          # Ignores venv, .env, chroma_db

---

## Key Concepts Learned

| Concept | What It Means |
|---------|--------------|
| Embeddings | Numbers representing meaning — similar text = similar vectors |
| Vector database | Searches by similarity not exact match |
| Chunking | Split docs into pieces for precise retrieval |
| Chunk overlap | Shared text between chunks to avoid losing context |
| RAG | Ground LLM answers in real documents to reduce hallucination |
| Source citations | Show which chunks were used to generate the answer |

---

## Author

**Richardy Lobo' Sapan**
- GitHub: [@richardy-lobo-sapan](https://github.com/richardy-lobo-sapan)
- LinkedIn: [richardylobosapan](https://www.linkedin.com/in/richardylobosapan/)
