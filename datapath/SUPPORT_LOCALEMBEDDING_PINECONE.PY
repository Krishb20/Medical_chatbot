from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ====== CONFIG ======
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # e.g., "us-east-1"
INDEX_NAME = "midebot"

# ====== 1. Load PDFs ======
def load_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

documents = load_files("datapath/")
print(f"📂 Total documents loaded: {len(documents)}")

# ====== 2. Create chunks ======
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    return splitter.split_documents(docs)

chunks = create_chunks(documents)
print(f"✂️ Total chunks created: {len(chunks)}")

# ====== 3. Initialize Pinecone ======
pc = Pinecone(api_key=PINECONE_API_KEY)

# HuggingFace embedding model (BYOE)
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",   # Or "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs={"device": "cpu"},     # set "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": True}
)

# Detect embedding dimension automatically
test_vec = embedding_model.embed_query("Hello world")
embedding_dim = len(test_vec)
print(f"🔢 HuggingFace embedding dimension: {embedding_dim}")

# Create Pinecone index if it doesn’t exist
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=embedding_dim,  # must match HuggingFace model dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"✅ Created new Pinecone index: {INDEX_NAME}")
else:
    print(f"ℹ️ Using existing Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# ====== 4. Store vectors into Pinecone ======
docs = [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

db = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=INDEX_NAME
)

print("🎉 Documents embedded with HuggingFace and stored in Pinecone!")
