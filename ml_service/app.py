from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

model = None

class TextRequest(BaseModel):
    texts: List[str]

def get_model():
    global model
    if model is None:
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return model

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/embed")
def embed_texts(req: TextRequest):
    current_model = get_model()
    processed_texts = ["passage: " + t for t in req.texts]
    embeddings = current_model.encode(
        processed_texts,
        normalize_embeddings=True
    ).tolist()
    return {"embeddings": embeddings} 