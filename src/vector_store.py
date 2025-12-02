import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    return pc


def create_or_load_index(pc, index_name="medical-chatbot"):
    exists = pc.has_index(index_name)

    if not exists:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
    return pc.Index(index_name)


def create_vector_store(docs, embedding, index_name="medical-chatbot"):
    return PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
        index_name=index_name
    )
