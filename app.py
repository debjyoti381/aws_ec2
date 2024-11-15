import os
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
from langchain_core.documents import Document

# Step 1: Load environment variables
load_dotenv()

# Set environment variable for Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyARxm0Lk5SXSHRMt_Rw3iklQrVQcGRgVCA"

# Step 2: Load multiple CSV files from local folder
def get_combined_csv_text_from_local(folder_path):
    """
    Fetch all CSV files from the specified local folder,
    combine their content into a single string.

    Args:
        folder_path (str): Path to the local folder containing CSV files.

    Returns:
        str: Combined text of all CSV files.
    """
    all_text = []

    # Iterate over all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            print(f"Processing file: {file_name}")
            file_path = os.path.join(folder_path, file_name)

            # Read CSV content into pandas DataFrame and convert to text
            try:
                df = pd.read_csv(file_path)
                all_text.append(df.to_string(index=False))
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Combine all text
    return "\n".join(all_text)


# Step 3: Split text into manageable chunks and create Document objects
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"id": str(i)}) for i, chunk in enumerate(chunks)]


# Step 4: Set up the FAISS vector store with embeddings
def setup_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# Step 5: Set up the RAG chain with Google Gemini API
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


# Step 6: Handle the query and return an answer
def answer_question(chain, query):
    answer = chain.invoke(query)
    return answer


# --- Main Execution ---

# Path to the local folder containing CSV files
local_folder_path = "./csv_files"

# Step 1: Load and process CSV data from local folder
raw_text = get_combined_csv_text_from_local(local_folder_path)
if not raw_text.strip():
    raise ValueError("No valid CSV content was found.")

docs = get_text_chunks(raw_text)

# Step 2: Set up the FAISS vector store
vectorstore = setup_vectorstore(docs)

# Step 3: Set up the Retrieval-Augmented Generation (RAG) chain
chain = setup_rag_chain(vectorstore)

# Step 4: Query the chatbot
print("System Ready. You can now ask questions about the CSV data.\n")
while True:
    query = input("Enter your question (type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    answer = answer_question(chain, query)
    print(f"Answer: {answer}")
