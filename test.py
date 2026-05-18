from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:\\Users\\A\\Downloads\\1706.03762v7.pdf')
pages = loader.load()
print(f'Loaded {len(pages)} pages')
print(pages[0].page_content[:200])