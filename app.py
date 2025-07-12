import streamlit as st
from utils import get_pdf_text, get_text_chunks, get_vector_store, get_ollama_rag_chain
import os
from langchain_community.llms import Ollama

OLLAMA_BASE_URL = "http://localhost:11434" 
OLLAMA_MODEL = "mistral"

def main():
    pdf_docs = st.file_uploader(
        "Upload your PDF Files here", accept_multiple_files=True, type="pdf"
    )

    if pdf_docs:
        with st.spinner("Processing PDFs... This may take a moment."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.vector_store = vector_store
        st.success("PDFs processed and ready to query!")
        st.info(f"Loaded {len(text_chunks)} text chunks.")

    if "vector_store" in st.session_state:
        user_question = st.text_input("Ask a question about the items and prices:")

        if user_question:
            try:
                _ = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL).invoke("hi", temperature=0)
            except Exception as e:
                st.error(f"Could not connect to Ollama server at {OLLAMA_BASE_URL}. Is Ollama running and the '{OLLAMA_MODEL}' model pulled? Error: {e}")
                st.stop()

            retrieval_chain = get_ollama_rag_chain(st.session_state.vector_store)

            with st.spinner("Finding answer..."):
                response = retrieval_chain.invoke({"input": user_question})
                print(response)
                st.write("---")
                st.subheader("Answer:")
                st.write(response["answer"])
                st.write("---")
    else:
        st.warning("Please upload PDF(s) to enable questioning.")

if __name__ == "__main__":
    main()