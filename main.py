from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from IPython.display import display, Markdown
from settings import init


def main():
    init()
    st.set_page_config(page_title="Q & A on your data", page_icon="📝")
    st.header("Ask your PDF 💬")
    st.markdown(
        """
        # Welcome to Streamlit! 👋
        """
    )
    st.sidebar.success("Select a demo above.")

    split_docs = None

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    split_docs = split_pdf(pdf, split_docs)

    measure_cost_of_embeddings(split_docs)

    faiss_vector_store = create_faiss_vector_store(split_docs)

    llm = setup_llm()

    compression_retriever = setup_contextual_compression_retriever(faiss_vector_store, llm)

    # show user input
    user_question = st.text_input("What do you want to know from your content?")
    vector_search_results = None
    if user_question:
        vector_search_results = compression_retriever.get_relevant_documents(user_question)
        display_relevant_doc_search_results(vector_search_results)
    else:
        st.write("Please enter a question to search for.")

    st.write(vector_search_results)


def split_pdf(pdf, split_docs):
    if pdf is not None:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        split_docs = split_into_chunks(pages)
        print(f'Number of chunks of data: {len(split_docs)}')
    else:
        st.write("Please upload a PDF file.")
    return split_docs


def setup_langchain(chain_type_kwargs, compression_retriever, llm):
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain


def print_result(user_question, result):
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


def setup_prompt_template():
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
    return prompt


def display_relevant_doc_search_results(vector_search_results):
    line_separator = "\n"
    display(Markdown(f"""
            ## Search results:{line_separator}
            {line_separator.join([
        f'''
              ### Source:{line_separator}{r[0].metadata['source']}{line_separator}
              #### Score:{line_separator}{r[1]}{line_separator}
              #### Content:{line_separator}{r[0].page_content}{line_separator}
              '''
        for r in vector_search_results
    ])}
            """))


def setup_contextual_compression_retriever(faiss_vector_store, llm):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=faiss_vector_store.as_retriever()
    )
    return compression_retriever


def setup_llm():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)
    return llm


def create_faiss_vector_store(split_docs):
    embeddings = OpenAIEmbeddings()
    faiss_vector_store = FAISS.from_documents(split_docs, embeddings)
    return faiss_vector_store


def measure_cost_of_embeddings(split_docs):
    # create a GPT-4 encoder instance
    enc = tiktoken.encoding_for_model("gpt-4")
    total_word_count = sum(len(doc.page_content.split()) for doc in split_docs)
    total_token_count = sum(len(enc.encode(doc.page_content)) for doc in split_docs)
    print(f"\nNumber of words: {total_word_count}")
    print(f"\nEstimated Number of tokens: {total_token_count}")
    print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")


def split_into_chunks(pages):
    chunk_size_limit = 1000
    max_chunk_overlap = 0
    separator = "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap)
    split_docs = text_splitter.split_documents(pages)
    return split_docs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
