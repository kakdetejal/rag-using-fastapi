import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag import load_pdf, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

@app.get("/")
def home():
    return {"message": "Welcome to the RAG API. RAG backend is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = load_pdf(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {exc}"
        ) from exc

    return {
    "message": f"File '{file.filename}' uploaded and processed successfully. Number of chunks created: {chunks}"
}

@app.post("/ask")
async def ask(data: dict):

    question = data.get("question", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        answer = ask_question(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate answer: {exc}"
        ) from exc

    return {"answer": answer}