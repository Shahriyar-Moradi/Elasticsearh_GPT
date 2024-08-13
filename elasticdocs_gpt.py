import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch,helpers
import numpy as np
import csv
import openai
import os
from typing import List
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
# openai.api_key = "sk-proj-n2gSbdt3L5MZVfdhNDtoT3BlbkFJW39ouvAFUqzMZRa4AqdE"
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
# openai_embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

template="""
You are an assistant for question-answering tasks.you should answer the  first in general and then in details base on provided context in the list.
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
from dotenv import load_dotenv
load_dotenv()

openai_embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# loaded_embeddings = np.load('embedded_data_filtered_test1.npy',allow_pickle=True)
# data=loaded_embeddings

cloud_id ="344d890d63df49d49082e1d70ca3d5b9:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGE3YTAxMzkxZTRmYjQ5OWNhYjA0NzExYTdlOTYwYjc3JGZiZGU3NjdlZTg5ZjQxMGRiOThmY2U0OTkzZjFhYWZk",

# openai.api_key = os.getenv("OPENAI_API_KEY")
import os
from openai import OpenAI
model = "gpt-4o-mini"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def es_connect():
    # es  = Elasticsearch("http://localhost:9200")
    # es = Elasticsearch(es_api_key="bjV3WUo1RUJYa1ZnQmsydm1sRUc6ZlpsZ3BtVHBUX096RTFpa1NoNFhLZw==",es_cloud_id="344d890d63df49d49082e1d70ca3d5b9:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGE3YTAxMzkxZTRmYjQ5OWNhYjA0NzExYTdlOTYwYjc3JGZiZGU3NjdlZTg5ZjQxMGRiOThmY2U0OTkzZjFhYWZk",
    #      )
    es = Elasticsearch(
  "https://a7a01391e4fb499cab04711a7e960b77.us-central1.gcp.cloud.es.io:443",
  api_key="bjV3WUo1RUJYa1ZnQmsydm1sRUc6ZlpsZ3BtVHBUX096RTFpa1NoNFhLZw==",
  # cloud_id=
)
    return es

es  = Elasticsearch("http://localhost:9200")

es=Elasticsearch(
        api_key="bjV3WUo1RUJYa1ZnQmsydm1sRUc6ZlpsZ3BtVHBUX096RTFpa1NoNFhLZw==",
        cloud_id="344d890d63df49d49082e1d70ca3d5b9:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGE3YTAxMzkxZTRmYjQ5OWNhYjA0NzExYTdlOTYwYjc3JGZiZGU3NjdlZTg5ZjQxMGRiOThmY2U0OTkzZjFhYWZk",
)

vector_db=ElasticsearchStore(
    es_api_key="bjV3WUo1RUJYa1ZnQmsydm1sRUc6ZlpsZ3BtVHBUX096RTFpa1NoNFhLZw==",
    es_cloud_id="344d890d63df49d49082e1d70ca3d5b9:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGE3YTAxMzkxZTRmYjQ5OWNhYjA0NzExYTdlOTYwYjc3JGZiZGU3NjdlZTg5ZjQxMGRiOThmY2U0OTkzZjFhYWZk",
    embedding=openai_embedding,
        # embedding=model_ST.encode(docs),
    index_name="test1",
)


def generate_embedding(text):
    
    text = text.replace("\n", " ")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-ada-002", chunk_size=500)
    # return client.embeddings.create(model="text-embedding-ada-002",input = [text]).data[0].embedding
    return embeddings.embed_documents([text])[0]

def create_index(index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    # "NoticeId": {"type": "keyword"},
                    "Title": {"type": "text"},
                    "Description": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 1536}  # Change dims according to the embedding size
                }
            }
        },
        ignore=400  # Ignore index already exists error
    )

def index_data(index_name, data):
    actions = [
        {
            "_index": index_name,
            # "_id": row["NoticeId"],
            "_source": {
                # "NoticeId": row["NoticeId"],
                "Title": row["Title"],
                "Description": row["Description"],
                "embedding": row["embedding"]
            }
        }
        for row in data
    ]
    helpers.bulk(es, actions)



# def search(query_text):
#     # cid = os.environ['cloud_id']
#     # cp = os.environ['cloud_pass']
#     # cu = os.environ['cloud_user']
#     es = es_connect()

#     # Elasticsearch query (BM25) and kNN configuration for hybrid search
#     query = {
#         "bool": {
#             "must": [{
#                 "match": {
#                     "title": {
#                         "query": query_text,
#                         "boost": 1
#                     }
#                 }
#             }],
#             "filter": [{
#                 "exists": {
#                     "field": "title-vector"
#                 }
#             }]
#         }
#     }

#     knn = {
#         "field": "title-vector",
#         "k": 1,
#         "num_candidates": 20,
#         "query_vector_builder": {
#             "text_embedding": {
#                 "model_id": "sentence-transformers__all-distilroberta-v1",
#                 "model_text": query_text
#             }
#         },
#         "boost": 24
#     }

#     fields = ["title", "body_content", "url"]
#     index = 'test1'
#     resp = es.search(index=index,
#                      query=query,
#                      knn=knn,
#                      fields=fields,
#                      size=1,
#                      source=False)

#     body = resp['hits']['hits'][0]['fields']['body_content'][0]
#     url = resp['hits']['hits'][0]['fields']['url'][0]

#     return body, url




index_name = 'test1'
# create_index(index_name)
# index_data(index_name, data)


def search(query):
    query_embedding = generate_embedding(query)
    
    script_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {
                    "query_vector": query_embedding
                }
            }
        }
    }
#     script_query = {
#     "script_score": {
#         "query": {"match_all": {}},
#         "script": {
#             "source": "cosineSimilarity(params.query_vector, 'abs_emb') + 1.0",
#             "params": {"query_vector": query_embedding}
#         }
#     }
# }
    
    response = es.search(
        index=index_name,
        request_timeout=60,
        body={
            "size": 4,
            "query": script_query
        }
    )
    
    return response['hits']['hits']


def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    # response = openai.ChatCompletion.create(model=model,
    #                                         messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    response = client.chat.completions.create(
     messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}],
    # model="gpt-4o",
    model="gpt-4o-mini",
    # messages=[
    #     {
    #         "role": "user",
    #         "content": "Say this is a test",
    #     }
    # ],
)


    # print('reponse : ',response.choices[0].message.content)
    print('--------------------------------------------------')
    return response.choices[0].message.content

    # return response["choices"][0]["message"]["content"]


st.title("GOV GPT")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")


negResponse = "answer with your general knowledge and then mention information is not provided"
if submit_button:
    
    # responses = search(query)
    
#     results = vector_db.similarity_search(
#     query=query,
#     k=2,
#     # filter=[{"term": {"metadata.source.keyword": "Title"}}],
# )
    retriever=vector_db.as_retriever()
    prompt=ChatPromptTemplate.from_template(template)
    llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini",temperature=1)
    rag_chain=(
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# query="construction contract"
    response=rag_chain.invoke(query)

# print(response)
    # print('\nresults:\n',results)
    # for res in results:
    #     print('results',f"{res.page_content}")
    # for doc, score in results:
    #     print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
    
    prompt = f"Answer this question: {query}. firstly just very short answer base on your general knowledge not in details and don't mention short or first answer, then \nUsing only the information from this Elastic Doc: {response}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
    answer = chat_gpt(prompt)
    # print('answer : ',answer)
    
    if negResponse in answer:
        st.write(f"GOVGPT: {answer.strip()}")
    else:
        st.write(f"GovGPT: {answer.strip()}\n\n Doc: {response}")
        # for res in results:
        #     print('results',f"{res.page_content} [0[0]")
            # st.write(f"ChatGPT: {answer.strip()}\n\nDocsTitle: {res.page_content}[0][0][0]")
        # for result in responses:
            # print()
            # print(f":Title : {result['_source']['Title']}")
            # print(f"Description: {result['_source']['Description']}")
            # print("------")
            # st.write(f"ChatGPT: {answer.strip()}\n\nDocsTitle: {result['_source']['Title']}")
            # st.write(f"Title : {result['_source']['Title']}")         
            # st.write(f"ChatGPT: {answer.strip()}\n\nDocs: Description:{result['_source']['Description']}")
