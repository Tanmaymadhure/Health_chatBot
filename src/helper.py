import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# LOAD ALL PDF FILES
# -----------------------------
def load_pdf(data_path: str):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# -----------------------------
# FILTER OUT VERY SHORT PAGES
# -----------------------------
def filter_short_documents(docs: List[Document]) -> List[Document]:
    filtered = []
    for d in docs:
        text_clean = d.page_content.replace("\n", " ").strip()
        if len(text_clean.split()) > 20:
            src = d.metadata.get("source", "")
            filtered.append(Document(page_content=text_clean, metadata={"source": src}))
    return filtered


# -----------------------------
# SPLIT INTO TEXT CHUNKS
# -----------------------------
def chunk_documents(min_docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=25
    )
    return splitter.split_documents(min_docs)


# -----------------------------
# EMBEDDINGS
# -----------------------------
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return embedding
