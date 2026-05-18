import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

dimension = 384

## Create a FAISS index
index = faiss.IndexFlatL2(dimension)

document_chunks = []

def create_embeddings(text_chunks):

    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)

    return np.array(embeddings).astype('float32')

def store_embeddings(text_chunks, embeddings):
    index.add(embeddings)

    document_chunks.extend(text_chunks)
    print(f"Added {len(text_chunks)} chunks to the index. Total chunks in index: {len(document_chunks)}")
    print(f"Current document chunks: {document_chunks}")

def search_similar_chunks(question,k=3):

    if not document_chunks:
        raise ValueError("No documents have been uploaded yet.")

    question_embedding = embedding_model.encode([question])

    result_count = min(k, len(document_chunks))

    distances, indices = index.search(np.array(question_embedding).astype('float32'), k=result_count)

    retrieved_chunks = []

    for i in indices[0]:
        if i >= 0:
            retrieved_chunks.append(document_chunks[i])

    return retrieved_chunks