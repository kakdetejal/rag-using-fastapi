# RAG PDF Chatbot using FastAPI and Streamlit

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on the document content.

The application uses FastAPI for the backend API development and Streamlit for the interactive frontend interface. PDF documents are processed using LangChain, converted into embeddings using Sentence Transformers, and stored in a FAISS vector database for semantic search. Relevant document chunks are retrieved and passed to the Groq LLM to generate accurate answers.

## Features
- Upload and process PDF documents
- Ask questions from uploaded PDFs
- Semantic search using FAISS
- AI-generated answers using Groq LLM
- Interactive UI with Streamlit
- REST API backend with FastAPI

## Technologies Used
- Python
- FastAPI
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Groq API

## Workflow
1. Upload PDF document
2. Extract and split text into chunks
3. Generate embeddings for chunks
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks based on user query
6. Generate contextual answer using LLM
