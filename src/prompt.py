from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
You are Medibot, an AI Medical Assistant.
Provide medical answers clearly, concisely, and safely.
Use only the given context. If you don’t know, say you don’t know.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Question: {input}\n\nContext:\n{context}")
])
