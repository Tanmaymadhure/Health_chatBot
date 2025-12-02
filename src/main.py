from src import (
    load_pdf,
    filter_short_documents,
    chunk_documents,
    download_embeddings,
    init_pinecone,
    create_or_load_index,
    create_vector_store,
    build_rag_chain
)

def main():
    print("🚀 Starting Medical RAG Pipeline")

    data_path = "Data"   # Folder with your PDFs

    # 1. Load PDFs
    docs = load_pdf(data_path)

    # 2. Filter short ones
    clean_docs = filter_short_documents(docs)

    # 3. Split into chunks
    chunks = chunk_documents(clean_docs)

    # 4. Embeddings
    embedding = download_embeddings()

    # 5. Pinecone
    pc = init_pinecone()
    index = create_or_load_index(pc, index_name="medical-chatbot")

    # 6. Vector Store
    vector_store = create_vector_store(chunks, embedding, index_name="medical-chatbot")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 7. RAG chain
    rag_chain = build_rag_chain(retriever)

    # 8. Example Query
    question = "What are symptoms of diabetes?"
    response = rag_chain.invoke({"input": question})

    print("\n🩺 Answer: ")
    print(response["answer"])


if __name__ == "__main__":
    main()
