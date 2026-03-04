"""
Contains the structured ChatPromptTemplate for strict Retrieval-Augmented Generation (RAG).
Enforces:
1. Answering strictly from context.
2. A required fallback string if the answer isn't found.
3. Mandatory page number citations.
"""

from langchain_core.prompts import ChatPromptTemplate

QA_SYSTEM_TEMPLATE = """You are a helpful, accurate, and highly strict assistant.
Your task is to answer the user's question using ONLY the provided document context.

Follow these rules exactly:
1. You must base your answer exclusively on the provided Context. Do not use outside knowledge.
2. If the Context does not contain the information needed to answer the question, you MUST return EXACTLY this string, with no other text: "I could not find an answer to that in the document."
3. Provide your answer clearly and naturally. Do NOT append page citations (like "(Page 1)") inside your text output.

Context information:
{context}
"""

QA_USER_TEMPLATE = """Question: {question}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_TEMPLATE),
    ("user", QA_USER_TEMPLATE)
])
