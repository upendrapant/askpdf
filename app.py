import os
import tempfile
import streamlit as st

from rag.loader import load_and_split
from rag.embedder import embed_and_store, clear_chroma_collection, delete_documents_by_source, get_vectorstore
from rag.retriever import retrieve_and_answer

#Streamlit Page Configuration
st.set_page_config(page_title="AskPDF", page_icon="📚", layout="wide")

#Initialize session state for caching
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

st.title("💭 Ask your PDF")

st.sidebar.title("📂 Upload PDFs")

# New Session Button
if st.sidebar.button("New Session"):
    clear_chroma_collection()
    st.session_state.vectorstore = get_vectorstore()
    st.session_state.processed_files = set()
    st.session_state.messages = []
    st.session_state.file_uploader_key += 1
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF documents", 
    type=["pdf"], 
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

# Determine files that were removed
current_uploaded_filenames = {f.name for f in (uploaded_files or [])}
files_to_remove = st.session_state.processed_files - current_uploaded_filenames

if files_to_remove:
    for file_name in files_to_remove:
        delete_documents_by_source(file_name)
        st.session_state.processed_files.remove(file_name)
        st.toast(f"Removed data for `{file_name}`.", icon="🗑️")
    
    # If all files are removed, we can just clear chat
    if not st.session_state.processed_files:
        st.session_state.messages = []
        st.toast("All documents cleared.", icon="🗑️", duration=10)

# Process new files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                with st.sidebar.spinner(f"Processing `{uploaded_file.name}` (parsing and embedding)..."):
                    # Save the uploaded file to a temporary file because loader.py expects a file path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Process the PDF using our RAG pipeline
                    try:
                        docs = load_and_split(tmp_path, original_filename=uploaded_file.name)
                        st.session_state.vectorstore = embed_and_store(docs)
                        st.session_state.processed_files.add(uploaded_file.name)
                        
                        st.toast(f"Successfully processed `{uploaded_file.name}` ({len(docs)} chunks indexed)!", icon="✅", duration=10)
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
            except Exception as e:
                st.error(f"Error processing `{uploaded_file.name}`: {str(e)}")
                st.toast(f"Error processing `{uploaded_file.name}`: {str(e)}", icon="❌")


# Chat Interface
st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources Context"):
                if isinstance(message["sources"], str):
                    st.info(f"Information gathered from: {message['sources']}")
                else:
                    for src in message["sources"]:
                        # Show filename as well
                        src_name = src.get("source", "Unknown")
                        st.markdown(f"**{src_name} - Page {src['page']}**\n> {src['text']}")

# Accept user input using Streamlit's chat input
is_ready = len(st.session_state.processed_files) > 0
prompt = st.chat_input("Ask a question about the document(s)...", disabled=not is_ready)

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve context and generate answer
                result = retrieve_and_answer(prompt, st.session_state.vectorstore)
                
                # Render the answer
                st.markdown(result["answer"])
                
                # Render sources in an expander
                if result["sources"]:
                    with st.expander("Sources Context"):
                        for src in result["sources"]:
                            src_name = src.get("source", "Unknown")
                            st.markdown(f"**{src_name} - Page {src['page']}**\n> {src['text']}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
