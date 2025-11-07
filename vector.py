import os
import shutil
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


CSV_PATH = "restaurant_reviews.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8")



embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


DB_DIR = "chroma_langchain_db"


if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)


documents = []
ids = []

for i, row in df.iterrows():
    text = f"{row.get('Title', '')} {row.get('Review', '')}".strip()
    metadata = {
        "rating": row.get("Rating", None),
        "date": row.get("Date", None)
    }
    documents.append(Document(page_content=text, metadata=metadata))
    ids.append(str(i))


vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=DB_DIR,
    embedding_function=embeddings
)


vector_store.add_documents(documents=documents, ids=ids)


retriever = vector_store.as_retriever(search_kwargs={"k": 5})
