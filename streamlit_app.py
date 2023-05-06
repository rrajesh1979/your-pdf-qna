import streamlit as st
from langchain.callbacks import get_openai_callback


def main():
    init()
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
        if pages is None:
            st.write("Please upload a PDF file.")
            return
        else:
            split_docs = split_into_chunks(pages)
            measure_cost_of_embeddings(split_docs)
            knowledge_base = create_faiss_vector_store(split_docs)
            llm = setup_llm()

            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                vector_search_results = knowledge_base.similarity_search(user_question)
                display_relevant_doc_search_results(vector_search_results)
                chain = setup_langchain(llm)
                with get_openai_callback() as cb:
                    result = chain.run(input_documents=vector_search_results, question=user_question)
                    print(cb)

                st.write(result)


def init():
    import os
    from os.path import join, dirname
    from dotenv import load_dotenv
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    print("Initialized...")


def read_pdf(pdf_file):
    from PyPDF2 import PdfReader
    pages = None
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        pages = [page for page in pdf_reader.pages]
        print(f'Number of pages in PDF: {len(pages)}')
    else:
        print(f'Please upload a PDF file.')
    return pages


def split_into_chunks(pages):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text = ""
    for page in pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    split_docs = text_splitter.create_documents([text])
    print(f'Number of chunks: {len(split_docs)}')
    return split_docs


def measure_cost_of_embeddings(split_docs):
    import tiktoken
    # create a GPT-4 encoder instance
    enc = tiktoken.encoding_for_model("gpt-4")
    total_word_count = sum(len(doc.page_content.split()) for doc in split_docs)
    total_token_count = sum(len(enc.encode(doc.page_content)) for doc in split_docs)
    print(f"\nNumber of words: {total_word_count}")
    print(f"\nEstimated Number of tokens: {total_token_count}")
    print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")


def create_faiss_vector_store(split_docs):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    embeddings = OpenAIEmbeddings()
    faiss_vector_store = FAISS.from_documents(split_docs, embeddings)
    print(f"Vector store created successfully.")
    return faiss_vector_store


def setup_llm():
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
    print(f"LLM setup successfully.")
    return llm


def display_relevant_doc_search_results(results):
    for result in results:
        print(f"Content:: {result.page_content[:100]}")
        print("\n")
    print(f"Relevant documents retrieved successfully.")


def setup_langchain(llm):
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm, chain_type="stuff")
    print(f"Langchain setup successfully.")
    return chain


if __name__ == '__main__':
    main()
