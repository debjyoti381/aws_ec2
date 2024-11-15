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

# Step 2: Load multiple CSV files from AWS S3 and convert them to a combined string
def get_combined_csv_text_from_s3(bucket_name, prefix):
    """
    Fetch all CSV files from the specified S3 bucket and prefix,
    combine their content into a single string.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Prefix or folder path in the bucket.

    Returns:
        str: Combined text of all CSV files.
    """
    s3 = boto3.client('s3')
    all_text = []

    # List objects in the bucket with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        raise FileNotFoundError(f"No files found in bucket '{bucket_name}' with prefix '{prefix}'.")

    # Iterate over all objects
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.csv'):
            print(f"Processing file: {key}")
            # Fetch the file content
            csv_object = s3.get_object(Bucket=bucket_name, Key=key)
            csv_content = csv_object['Body'].read().decode('utf-8')

            # Read CSV content into pandas DataFrame and convert to text
            df = pd.read_csv(StringIO(csv_content))
            all_text.append(df.to_string(index=False))

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

# S3 bucket and prefix for CSV files
s3_bucket_name = "datasetschatbot"  # Replace with your S3 bucket name
s3_prefix = "csv_files/"  # Replace with your folder prefix

# Step 1: Load and process CSV data from S3
raw_text = get_combined_csv_text_from_s3(s3_bucket_name, s3_prefix)
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
