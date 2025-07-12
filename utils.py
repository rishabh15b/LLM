from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF file objects."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks using Ollama embeddings."""
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_ollama_rag_chain(vector_store):
    """Sets up the conversational RAG chain using Ollama LLM and modern LangChain."""
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0.7)

    qa_prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context only.
    If the answer is not in the provided context, just say, "The information for this item is not found in the uploaded documents."
    Do not add any additional information that is not directly derivable from the context.
    I will be uploading a conversation with a customer who is requesting for the product. Look at provided price list check for minimum and maximum price. Your task is to sell the product.
    Context:\n {context} \n
    Question:\n {input} \n

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    print ("Creating retriever from vector store...", document_chain)

    retriever = vector_store.as_retriever()

    print ("Creating retrieval chain with retriever and document chain...", retriever)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print ("Retrieval chain created successfully. final output ", retrieval_chain)

    return retrieval_chain