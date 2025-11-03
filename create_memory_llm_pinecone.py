from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.base import Embeddings
import os
import uuid

# ====== CONFIG ======
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # e.g., "us-east-1"
INDEX_NAME = "midebot"

# ====== 1. Load PDFs ======
def load_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

documents = load_files("datapath/")
print(f"Total documents loaded: {len(documents)}")

# ====== 2. Create chunks ======
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    return splitter.split_documents(docs)

chunks = create_chunks(documents)
print(f"Total chunks created: {len(chunks)}")

# ====== 3. Initialize Pinecone ======
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"Created new Pinecone index: {INDEX_NAME}")
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# ====== 4. Define a LangChain-compatible embedding class ======
class LlamaTextEmbed(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            r = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[t],
                parameters={"dimension": 1024, "input_type": "passage"}
            )
            embeddings.append(r.data[0].values)
        return embeddings

    def embed_query(self, text):
        r = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={"dimension": 1024, "input_type": "passage"}
        )
        return r.data[0].values

embedding_model = LlamaTextEmbed()

# ====== 5. Store via PineconeVectorStore ======
docs = [Document(page_content=chunk.page_content) for chunk in chunks]

db = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=INDEX_NAME
)
print("✅ Documents added to Pinecone via PineconeVectorStore!")

# # ====== 6. (Optional) Direct upsert without LangChain ======
# for i, chunk in enumerate(chunks):
#     r = pc.inference.embed(
#     model="llama-text-embed-v2",
#     inputs=[chunk.page_content],
#     parameters={"dimension": 1024, "input_type": "passage"}
# )
#     vector = res.data[0].values
#     index.upsert([
#         (f"chunk-{i}", vector, {"text": chunk.page_content})
#     ])
# print("✅ Documents also added via direct Pinecone upsert!")
