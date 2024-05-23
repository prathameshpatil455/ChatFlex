import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv 
import asyncio

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf):
    with open(pdf, "rb") as file:
        pdf_reader = PdfReader(file)
        return ''.join([page.extract_text() for page in pdf_reader.pages])



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)
    


def get_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)




def process_pdf_files(pdf_files):
    raw_text = ''.join([get_pdf_text(pdf) for pdf in pdf_files])
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    get_vector_store(text_chunks, embeddings)
    st.success("PDF Processing Done!!!")



def get_data_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return ''.join([document.page_content for document in documents])


def process_urls(urls):
    combined_text = ''.join([get_data_from_url(url) for url in urls])
    text_chunks = get_text_chunks(combined_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    get_vector_store(text_chunks, embeddings)
    st.success("URL Processing Done!!!")



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat With Multiple PDF and URL")
    st.header("Chat with Multiple PDF and Websites using ChatFlex ")

    user_question = st.text_input("Ask a Question from the PDF Files or URL")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)


        if st.button("Submit & Process") and pdf_files:
            with st.spinner("Processing PDFs..."):
                process_pdf_files(pdf_files)

        st.title("Website URL:")
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}")
            if url:
                urls.append(url)

        if st.button("Process URLs") and urls:
            with st.spinner("Processing URLs..."):
                process_urls(urls)



if __name__ == "__main__":
    main()