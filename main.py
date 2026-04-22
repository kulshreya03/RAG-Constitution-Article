from qdrant_client import QdrantClient
import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition,Range, PayloadSchemaType, HnswConfigDiff, MatchValue
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
load_dotenv()

from data import dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent


client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collection_name = "constitution"
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_SIZE = 384
EMBEDDING_MODEL= "all-MiniLM-L6-v2"

collections = client.get_collections().collections
names = [c.name for c in collections]

if collection_name not in names:
    client.create_collection(
        collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=32,              # acc
                ef_construct=200
            )
        )

#load data-> create text then embed
def create_text(example):
        return f"""
        {example['article_id']}
        
        {example['article_desc']}
        """.strip()

def insert_data():
    

    texts = [create_text(x) for x in dataset['train']]
    vector = model.encode(texts).tolist()

    '''
    payload = {
        "article_id": article_id
    }
    '''

    #Points
    points = []

    for i, (text, vector) in enumerate(zip(texts, vector)):
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "meta": {
                        "article_id": dataset['train'][i]["article_id"]
                    },
                    "data": {
                        "content": dataset['train'][i]["article_desc"]
                    }
                }
            )
        )

    client.upsert(
        collection_name="constitution",
        points=points
    )
    print(len(points))

def delete_collection():
    client.delete_collection(collection_name)



def search(query):
    #query = input("\nEnter query : ")
    query_vector = model.encode(query).tolist()

    result = client.query_points(
        collection_name=collection_name,
            query=query_vector,
            
    #        query_filter= Filter(
    #           must=[
    #              FieldCondition(
    #                 key="meta.",
        #                range=Range(lte=5)
        #           )
        #      ]
        # ),
            
            limit=3,
            search_params=models.SearchParams(hnsw_ef=500)
    )

    contexts = []
    for point in result.points:
        content = point.payload["data"]["content"]
        article_id = point.payload["meta"]["article_id"]

        contexts.append(f"{article_id}: {content}")

    context_text = "\n\n".join(contexts)



    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash-lite",
        max_output_tokens = 200
    )

    agent = create_agent(
        model = llm,
        tools = [],
        system_prompt = """You are a legal assistant.

    Use the provided context to answer the question.
    - Do NOT hallucinate
    - If answer not in context, say "Not found" """
    )
    #response = llm.invoke({'context_text':context_text,'query':query})
    response = agent.invoke({
    "messages": [
                {
                    "role": "user",
                    "content": f"""
                    Context:
                    {context_text}

                    Question:
                    {query}
                    """
                }
                        ]
                    })

    print("Context:",context_text)
    print("Response:",response["messages"][-1].content)


while True:
        print("\n1. Insert all Data")
        print("2. Search by Query")
        print("3. Delete Collection")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            insert_data()

        elif choice == "2":
            while True:
                query = input("\nEnter query (or 'exit'): ")
                if query.lower() == "exit":
                    break
                search(query)

        elif choice == "3":
            delete_collection()

        elif choice == "4":
            break
