import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Set environment variable for Google API Key
# GOOGLE_API_KEY = "AIzaSyARxm0Lk5SXSHRMt_Rw3iklQrVQcGRgVCA"
os.environ["GOOGLE_API_KEY"] = "AIzaSyARxm0Lk5SXSHRMt_Rw3iklQrVQcGRgVCA"


# Load multiple CSV files and convert them to a combined string
def get_combined_csv_text(directory_path):
    all_text = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
            all_text.append(text)
    return "\n".join(all_text)

# Split text into manageable chunks and create Document objects
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"id": str(i)}) for i, chunk in enumerate(chunks)]

# Set up the FAISS vector store with embeddings
def setup_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Set up the RAG chain with Google Gemini API
def setup_rag_chain(vectorstore, model_name="gemini-1.5-flash"):
    template = """Answer the question in a single sentence with correct answer . Do not required extra explanation from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say, "answer is not available in the context.
    If user asks you like 1. who build you then give the answer as Utkarsh and 2. what is your name then give the answer as FlivoAI chatbot or 3. who are you just say FlivoAI Chatbot to solve your queries".
    
    
    
    Context: {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.01)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# Handle the query and return an answer
def answer_question(chain, query):
    answer = chain.invoke(query)
    return answer

# Streamlit interface for chatting with CSV data
def main():
    st.title("Chat - BOT")

    # Load CSV data and process it
    csv_directory = "csv_files"
    if "text_chunks" not in st.session_state:
        with st.spinner("Processing CSV files..."):
            raw_text = get_combined_csv_text(csv_directory)
            docs = get_text_chunks(raw_text)
            vectorstore = setup_vectorstore(docs)
            chain = setup_rag_chain(vectorstore)
            st.session_state['chain'] = chain
            st.session_state["text_chunks"] = docs

    # Query input
    if 'chain' in st.session_state:
        query = st.text_input("Ask a question about the CSV data:")
        
        if query:
            answer = answer_question(st.session_state['chain'], query)
            st.write(f"Answer: {answer}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
