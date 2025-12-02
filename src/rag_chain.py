from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from prompt import prompt


def build_rag_chain(retriever):
    llm = ChatOllama(model="phi3", temperature=0.0)

    answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, answer_chain)

    return rag_chain
