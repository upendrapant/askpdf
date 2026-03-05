"""
Handles the retrieval and generation aspect of the RAG pipeline.
Retrieves relevant documents, formats the context, queries Gemini,
and returns the final answer along with source pages.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts.qa_prompt import qa_prompt


def retrieve_and_answer(question: str, vectorstore: Chroma) -> Dict[str, Any]:
    load_dotenv()
    
    # 1. Retrieve top 4 chunks via similarity search
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    
    # 2. Extract sources and format context
    sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    
    for doc in retrieved_docs:
        # We explicitly rely on the metadata structured in loader.py
        page = doc.metadata.get("page", 0)
        source_file = doc.metadata.get("source", "Unknown")
        
        sources.append({
            "page": page,
            "source": source_file,
            "text": doc.page_content.strip()
        })
            
        # Bind the page number directly next to the chunk text
        context_parts.append(f"Text chunk from {source_file}, Page {page}:\n{doc.page_content}\n---")
        
    formatted_context = "\n\n".join(context_parts)
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3
    )
    
    chain = qa_prompt | llm
    
    response = chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    answer_text = response.content.strip()
    
    # If the model hit the exact fallback string, we logically clear the sources
    fallback_string = "I could not find an answer to that in the document."
    if fallback_string in answer_text:
        sources = []
        

    return {
        "answer": answer_text,
        "sources": sources
    }
