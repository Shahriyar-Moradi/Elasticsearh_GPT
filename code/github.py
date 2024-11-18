from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

import os
import streamlit as st

from elasticsearch import Elasticsearch,helpers
import numpy as np
import csv
import openai
from openai import OpenAI
from typing import List
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv() 
openai_embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# model = "gpt-4o-mini"

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )
es_api_key = os.getenv('ES_API_KEY')
es_cloud_id = os.getenv('ES_CLOUD_ID')

vector_db=ElasticsearchStore(
      es_api_key = os.getenv('ES_API_KEY'),
    es_cloud_id = os.getenv('ES_CLOUD_ID'),
    embedding=openai_embedding,
    index_name="test1",
)
#Return vectorstore for the URL
def get_vector_store():
    vector_db=ElasticsearchStore(
      es_api_key = os.getenv('ES_API_KEY'),
    es_cloud_id = os.getenv('ES_CLOUD_ID'),
    embedding=openai_embedding,
    index_name="test1",
)
    return vector_db
#   loader = WebBaseLoader(url)
#   data = loader.load()

#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#   chunks = text_splitter.split_documents(data)
#   vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
  
#   return vector_store

#Returns history_retriever_chain
def get_retreiver_chain():
  llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini",temperature=0.5)
  retriever = vector_db.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])
  history_retriver_chain = create_history_aware_retriever(llm,retriever,prompt)

  return history_retriver_chain

#Returns conversational rag
def get_conversational_rag(history_retriever_chain):
  llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini",temperature=0.5)
  answer_prompt=ChatPromptTemplate.from_messages([
      ("system","Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)

  #create final retrieval chain
  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain,document_chain)

  return conversational_retrieval_chain

#Returns th final response
def get_response(user_input):
  history_retriever_chain = get_retreiver_chain(st.session_state.vector_db)
  conversation_rag_chain = get_conversational_rag(history_retriever_chain)
  response = conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_input
    })
  return response["answer"]



#Streamlit app

st.header("Chat with websites")

chat_history=[]
vector_store=[]


# Sidebar
# URL pasting in sidebar on the left
with st.sidebar:
  st.header("Paste your URL")
  website_url = st.text_input("Enter URL")

if website_url is None or website_url.strip()=="":
  st.info("Please enter a website URL")
else:
  #session state
  if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="I am a bot, how can I help you?")
    ]
   #create conversation chain
  if vector_store not in st.session_state:
      st.session_state.vector_db = get_vector_store()

  user_input=st.chat_input("Type your message here...")
  if user_input is not None and user_input.strip()!="":
    response = get_response(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

  for message in st.session_state.chat_history:
      if isinstance(message,AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      else:
        with st.chat_message("Human"):
          st.write(message.content)