import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from groq import Groq

from backend.vector_store import (
    create_embeddings,
    store_embeddings,
    search_similar_chunks
)


## env files accessibile here
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    text_chunks = []

    for chunk in chunks:
        text_chunks.append(chunk.page_content)

    embeddings = create_embeddings(text_chunks)

    store_embeddings(text_chunks, embeddings)

    return len(text_chunks)

def ask_question(question):

    retrieved_chunks = search_similar_chunks(question)

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question based on the context provided.

    Context:
    {context}

    User Question:
    {question}

    Answer:
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", 
             "content": prompt
             }
        ]
    )

    return response.choices[0].message.content