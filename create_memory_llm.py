
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



#1 load medical pdf
data_path = "datapath/"
def load_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
documents = load_files(data = data_path)
print(len(documents))
#2 create chunks from pdf
def create_chunk(extracteddata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =400,chunk_overlap = 40)
    chunks = text_splitter.split_documents(extracteddata)
    return chunks
chunks = create_chunk(extracteddata=documents)
print(len(chunks))

#3 create embedding for chunks
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
load_model = get_embedding_model()
#4 store embedding in FAISS Data base store
DB_Faiss_path = "vectorestore/db_faiss"
db = FAISS.from_documents(chunks, load_model)
db.save_local(DB_Faiss_path)