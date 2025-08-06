import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load your CSV file containing question-answer pairs
df = pd.read_csv("data/Question_Answer.csv")

# Convert each row to a Document object with metadata
documents = [
    Document(page_content=row["answer"], metadata={"question": row["question"]})
    for _, row in df.iterrows()
]

# Use a lightweight, CPU-friendly embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS vectorstore from the documents and embeddings
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the vectorstore locally for later use
vectorstore.save_local("faiss_index")

print("✅ Vectorstore built and saved successfully.")
