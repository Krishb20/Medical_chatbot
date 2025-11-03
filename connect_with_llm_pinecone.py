import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Step 1: HuggingFace Token & Model Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't make up anything.
Only answer based on the given context.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Initialize Pinecone (new method)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "midebot"  # Your Pinecone index name

pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 4: Load Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="embaas/sentence-transformers-e5-large-v2")

# Step 5: Connect to Pinecone index
db = PineconeVectorStore(index=pc.Index(INDEX_NAME), embedding=embedding_model)

# Step 6: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 7: Run Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])

