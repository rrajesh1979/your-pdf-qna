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
            compression_retriever = setup_contextual_compression_retriever(knowledge_base, llm)

            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                vector_search_results = compression_retriever.get_relevant_documents(user_question)
                display_relevant_doc_search_results(vector_search_results)
                prompt = setup_prompt_template()
                chain_type_kwargs = {"prompt": prompt}
                chain = setup_langchain(chain_type_kwargs, compression_retriever, llm)
                with get_openai_callback() as cb:
                    result = chain(user_question)
                    response = format_result(user_question, result)
                    print(cb)

                st.write(response)


def init():
    import os
    from os.path import join, dirname
    from dotenv import load_dotenv
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    print("Initialized...")


def read_pdf(pdf_file):
    from langchain.document_loaders import PyPDFLoader
    pages = None
    if pdf_file is not None:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        print('PDF loaded successfully.')
        print(f'Number of pages: {len(pages)}')
    else:
        print(f'Please upload a PDF file.')
    return pages


def split_into_chunks(pages):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    chunk_size_limit = 1000
    max_chunk_overlap = 0
    separator = "\n"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_limit,
        chunk_overlap=max_chunk_overlap,
        separators=[separator]
    )
    split_docs = text_splitter.split_documents(pages)
    print(f'Number of chunks of data: {len(split_docs)}')
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


def setup_contextual_compression_retriever(faiss_vector_store, llm):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=faiss_vector_store.as_retriever()
    )
    print(f"Contextual compression retriever setup successfully.")
    return compression_retriever


def display_relevant_doc_search_results(results):
    for result in results:
        print(
            f""" Page:: {result.metadata['page']} \n 
             Metadata:: {result.metadata['source']} \n
              Content:: {result.page_content[:100]} \n
            """)
    print(f"Relevant documents retrieved successfully.")


def setup_prompt_template():
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    system_template = """Use the following pieces of context to answer the users question.
        Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", 
        use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    print(f"Prompt template setup successfully.")
    return prompt


def setup_langchain(chain_type_kwargs, compression_retriever, llm):
    from langchain.chains import RetrievalQAWithSourcesChain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    print(f"Langchain setup successfully.")
    return chain


def format_result(user_question, result):
    from IPython.display import display, Markdown
    output_text = f"""### Question: 
  {user_question}
  ### Answer: 
  {result['answer']}
  ### Sources: 
  {result['sources']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
    display(Markdown(output_text))
    return Markdown(output_text)


if __name__ == '__main__':
    main()
