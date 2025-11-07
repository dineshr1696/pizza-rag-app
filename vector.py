import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd


# Load your CSV (upload to Colab or mount Drive)
df = pd.read_csv("/content/restaurant_reviews.csv")

# Use Hugging Face embeddings API
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        text = f"{row['Title']} {row['Review']}"
        doc = Document(page_content=text, metadata={"rating": row["Rating"], "date": row["Date"]})
        documents.append(doc)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
