import os
import tempfile
import streamlit as st

from rag.loader import load_and_split
from rag.embedder import embed_and_store
from rag.retriever import retrieve_and_answer

#Streamlit Page Configuration
st.set_page_config(page_title="AskPDF", page_icon="📚", layout="wide")

#Initialize session state for caching
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💭 Ask your PDF")

st.sidebar.title("📂 Upload PDFs")
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    # Check if we've already processed this specific file
    if st.session_state.pdf_filename != uploaded_file.name or st.session_state.vectorstore is None:
        try:
            with st.sidebar.spinner("Processing PDF (parsing and embedding)..."):
                # Save the uploaded file to a temporary file because loader.py expects a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process the PDF using our RAG pipeline
                try:
                    docs = load_and_split(tmp_path)
                    vectorstore = embed_and_store(docs)
                    
                    # Store explicitly in session_state to prevent re-processing on keystrokes
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_filename = uploaded_file.name
                    st.session_state.messages = []  # Clear chat history on new upload
                    
                    st.toast(f"Successfully processed `{uploaded_file.name}` ({len(docs)} chunks indexed)!", icon = "✅",duration=10)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        except Exception as e:
            st.error(f"Error processing the PDF: {str(e)}")
            st.toast(f"Error processing the PDF: {str(e)}", icon = "❌")
            st.session_state.vectorstore = None
            st.session_state.pdf_filename = None

else:
    # Clear session state if the upload widget is cleared
    if st.session_state.vectorstore is not None:
        st.session_state.vectorstore = None
        st.session_state.pdf_filename = None
        st.session_state.messages = []  # Clear chat history
        st.toast("Document cleared.", icon = "🗑️", duration=10)


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
                        st.markdown(f"**Page {src['page']}**\n> {src['text']}")

# Accept user input using Streamlit's chat input
is_ready = st.session_state.vectorstore is not None
prompt = st.chat_input("Ask a question about the document...", disabled=not is_ready)

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
                            st.markdown(f"**Page {src['page']}**\n> {src['text']}")
                
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
