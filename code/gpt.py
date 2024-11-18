import os
import streamlit as st
from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the RAG prompt template
template = """
You are an assistant for question-answering tasks. First, provide a general answer and then elaborate based on the provided context.
If the context does not contain the answer, reply with "I don't know based on the provided context."

Context: {context}
Question: {question}
Answer:
"""

# Initialize Elasticsearch vector store
vector_db = ElasticsearchStore(
    es_api_key=os.getenv('ES_API_KEY'),
    es_cloud_id=os.getenv('ES_CLOUD_ID'),
    embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    index_name="test1",
)

# Initialize OpenAI client and retriever
retriever = vector_db.as_retriever()
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4", temperature=0.7)

# Define RAG chain
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit app title
st.title("Interactive RAG Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset chat session
if st.button("Clear Chat"):
    st.session_state.chat_history = []

# Chat interface
st.markdown("### Chat with GOV GPT")

# Display conversation history dynamically
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**GOV GPT:** {message['content']}")

# Input form for user query
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("Type your message:")
    submit_button = st.form_submit_button("Send")

# Process user input and generate response
if submit_button and query.strip():
    # Append user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Generate RAG-based response
    try:
        response = rag_chain.invoke({"context": retriever, "question": query})
        assistant_response = response.strip()
    except Exception as e:
        assistant_response = "I encountered an error while processing your request. Please try again later."

    # Append assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the new response
    st.markdown(f"**GOV GPT:** {assistant_response}")
