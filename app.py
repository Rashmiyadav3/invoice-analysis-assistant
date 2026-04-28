import streamlit as st
import os
from dotenv import load_dotenv

from document_processor import extract_text_from_pdf, get_text_chunks
from vector_store import create_vector_store
from chat_engine import get_conversational_chain, answer_user_question

load_dotenv()

st.set_page_config(page_title="Invoice Inspector RAG", layout="wide")

def main():
    st.title("Invoice Assistant Analysis")
    st.write("Upload an invoice and ask questions to extract financial details!")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.warning("Please add your GOOGLE_API_KEY to the .env file, or type it below:")
        api_key_input = st.text_input("Enter your API Key here:", type="password")
        if api_key_input:
            api_key = api_key_input
        else:
            st.stop()

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for uploading
    with st.sidebar:
        st.header("Upload Document")
        pdf_docs = st.file_uploader("Upload your Invoice (PDF format)", accept_multiple_files=False, type=["pdf"])
        
        if st.button("Process Invoice"):
            with st.spinner("Parsing and indexing text..."):
                if pdf_docs:
                    raw_text = extract_text_from_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(text_chunks, api_key)
                    st.success("Invoice Processed & Indexed!")
                else:
                    st.error("Please upload a PDF invoice first.")

    # Main Chat interface display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box
    user_question = st.chat_input("Ask a question about the uploaded invoice...")

    if user_question:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate LLM response
        if st.session_state.vector_store is not None:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    chain = get_conversational_chain(api_key)
                    response = answer_user_question(user_question, st.session_state.vector_store, chain)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Please process an invoice from the sidebar first.")

if __name__ == "__main__":
    main()
