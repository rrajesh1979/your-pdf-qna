from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Q&A over your own data", page_icon="📝")
    st.header("Q&A over your own data 💬")
    st.markdown(
        """
        # Welcome to AI enabled Q&A App! 👋
        """
    )

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pages = read_pdf(pdf)
        split_docs = split_into_chunks(pages)
        print(f'Number of chunks of data: {len(split_docs)}')

        knowledge_base = create_faiss_vector_store(split_docs)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


def read_pdf(pdf):
    pages = None
    if pdf is not None:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
    else:
        st.write("Please upload a PDF file.")
    return pages


def split_into_chunks(pages):
    chunk_size_limit = 1000
    max_chunk_overlap = 0
    separator = "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap)
    split_docs = text_splitter.split_documents(pages)
    return split_docs


def create_faiss_vector_store(split_docs):
    embeddings = OpenAIEmbeddings()
    faiss_vector_store = FAISS.from_documents(split_docs, embeddings)
    return faiss_vector_store


if __name__ == '__main__':
    main()
