import os
import streamlit as st
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Initialize session state
def init_session_state():
    # Initialize all session state variables at startup
    session_state_vars = {
        "messages": [{"role": "assistant", "content": "Hi! How can I assist with your government contract needs today?"}],
        "chat_history": [],
        "conversation_memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        ),
        "current_response": None,
    }
    
    for var, default_value in session_state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Cache the embedding model
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Cache the vector store
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    return ElasticsearchStore(
        es_api_key=os.getenv('ES_API_KEY'),
        es_cloud_id=os.getenv('ES_CLOUD_ID'),
        embedding=embeddings,
        index_name="test1",
    )

def format_chat_history(messages: List[Dict]) -> str:
    formatted_history = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_history.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted_history[-5:])  # Keep last 5 messages for context

class ContractAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
             model="gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )
        self.vector_store = get_vector_store()
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
    def get_response(self, question: str) -> Dict:
        try:
            # Get relevant documents
            docs = self.base_retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Format chat history
            chat_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current question
            
            # Create prompt
            prompt = PromptTemplate.from_template("""
            You are a government contract specialist assistant. Use the following information to answer the question.

            Chat History: {chat_history}
            Context: {context}
            Question: {question}
            
            Please provide a clear and detailed response.
            """)
            
            # Create chain
            chain = (
                {"context": lambda x: context, 
                 "question": lambda x: x,
                 "chat_history": lambda x: chat_history}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response with streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = ""
                
                for chunk in chain.stream(question):
                    response += chunk
                    response_placeholder.markdown(response + "▌")
                response_placeholder.markdown(response)
            
            return {
                "answer": response,
                "sources": docs
            }
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error. Please try rephrasing your question.",
                "sources": []
            }

def main():
    st.set_page_config(
        page_title="Government Contract Assistant",
        page_icon="📑",
        layout="wide"
    )
    
    # Initialize session state first
    init_session_state()
    
    st.title("📑 Government Contract Assistant")
    st.markdown("""
    Welcome! I can help you with:
    - Finding active government contracts
    - Searching for specific contract details
    - Analyzing award information
    - Understanding contract requirements
    """)

    # Initialize assistant
    assistant = ContractAssistant()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_input := st.chat_input("Ask about government contracts..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display response
        response = assistant.get_response(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        
        # Display sources if available
        if response["sources"]:
            with st.expander("View Source Documents"):
                for idx, doc in enumerate(response["sources"], 1):
                    st.markdown(f"**Source {idx}:**\n{doc.page_content}\n")

if __name__ == "__main__":
    main()
