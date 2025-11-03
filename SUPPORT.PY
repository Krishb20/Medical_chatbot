pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings (E5-large-v2, normalized, with E5 prefixes)
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

base = HuggingFaceEmbeddings(
    model_name="embaas/sentence-transformers-e5-large-v2",
    encode_kwargs={"normalize_embeddings": True}
)

class E5Embeddings(Embeddings):
    def __init__(self, base):
        self.base = base
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return self.base.embed_documents(texts)
    def embed_query(self, text):
        return self.base.embed_query(f"query: {text}")

embedding_model = E5Embeddings(base)

# Vector store / retriever
index = pc.Index(INDEX_NAME)  # Index must be 1024-dim, cosine
db = PineconeVectorStore(index=index, embedding=embedding_model)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),  # ensure max_new_tokens fix in load_llm
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCES:")
for i, d in enumerate(response["source_documents"], 1):
    print(f"#{i} meta={d.metadata}")
