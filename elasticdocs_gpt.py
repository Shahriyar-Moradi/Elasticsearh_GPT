import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
openai_embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Elasticsearch vector store setup
vector_db = ElasticsearchStore(
    es_api_key=os.getenv('ES_API_KEY'),
    es_cloud_id=os.getenv('ES_CLOUD_ID'),
    embedding=openai_embedding,
    index_name="test1",
)

# Define prompt template
prompt_template = """

You are an assistant for question-answering tasks.First, answer with your general knowledge and you should answer the  first in general and then in details base on provided context in the list.
Use the following pieces of retrieved context to answer the question. 
ou are a helpful assistant and a government contract specialist. As a government contract specialist with extensive expertise, your task is to analyze the information of the provided government contracts and respond to a user's inquiry. When crafting your response, please adhere to the following guidelines:
    
    Please list the contracts in the following format:
   1) Title: title of the contract
   2) Decription: Brief description about the contract
   3)country and city
   4) Vendor: Who is offering the contract
   5) Active: Whether it is active or not
   6) Award date (If Non-Active): date on which it was awarded
   7) Award amount (If Non-Active): Money for the contract
   8) Award number (If Non-Active): award number for the contract
   9) Link: link to the contract 


    Please Remember:
       1) If a user asks for awards details, don't tell them that you don't have the capability of delivering it. If there are no award details available, tell the user these contracts are active thus there are no award detail available.
       2) If a user akss for award details of similar contracts, only list those contracts which have award details in them.
       3) Your first priority should be to always display Active contracts. If not available or specified by user, then display non-active contracts

If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.8),
    retriever=vector_db.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": custom_prompt}
)

# Function to get response
def get_response(question):
    try:
        result = rag_chain({"query": question})
        return result["result"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit App
st.title("Government Contract Assistant")

# Chat functionality
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist with your government contract needs today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user query
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Fetching  ..."):
            response = get_response(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
