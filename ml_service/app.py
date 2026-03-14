# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from typing import List

# # 1️⃣ Create FastAPI app
# app = FastAPI()

# # 2️⃣ Load pretrained multilingual embedding model
# model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# # 3️⃣ Define request schema
# class TextRequest(BaseModel):
#     texts: List[str]

# @app.post("/embed")
# def embed_texts(req: TextRequest):
#     embeddings = model.encode(req.texts).tolist()
#     return {"embeddings": embeddings}



from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

# Create FastAPI app
app = FastAPI()

# Load the BETTER multilingual model (E5-large)
# This model provides superior performance for mixed languages (Hebrew/Arabic/English)
#model = SentenceTransformer(
 #   "intfloat/multilingual-e5-base"
#)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Define request schema
class TextRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed_texts(req: TextRequest):
    # Add "passage: " prefix (Required by E5 model for best performance in retrieval tasks)
    processed_texts = ["passage: " + t for t in req.texts]
    
    # Generate embeddings with normalization (normalize_embeddings=True)
    # This optimizes the vectors for cosine similarity calculations
    embeddings = model.encode(processed_texts, normalize_embeddings=True).tolist()
    
    return {"embeddings": embeddings}