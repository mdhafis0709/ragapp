import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import os


load_dotenv()

# Step 1: Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 2: Web scraping function
def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script.extract()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Failed to scrape URL: {str(e)}"

# Step 3: Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Step 4: Create FAISS vector store using HuggingFace embeddings
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Step 5: Setup conversational chain using Groq LLM
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "answer is not available in the context" — don't guess.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0.3
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("📌 **Reply:**", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF & Web", layout="wide")
    st.header("📄🕸️ RAG-based QA on PDF + Website using Groq", divider='rainbow')

    user_question = st.text_input("💬 Ask a question about the content (PDF or Web):")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📥 Upload & Scrape")

        # Upload PDFs
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        # Enter website URL
        url = st.text_input("Or enter a website URL to scrape content:")

        if st.button("Submit & Process"):
            with st.spinner("🔄 Processing content..."):
                combined_text = ""

                # Process PDFs
                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    combined_text += pdf_text + "\n"

                # Process URL
                if url:
                    web_text = get_text_from_url(url)
                    combined_text += web_text + "\n"

                if combined_text.strip() == "":
                    st.error("No valid content found to process.")
                else:
                    chunks = get_text_chunks(combined_text)
                    get_vector_store(chunks)
                    st.success("✅ Content processed and indexed successfully!")

if __name__ == "__main__":
    main()
