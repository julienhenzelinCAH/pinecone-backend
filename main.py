from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import uuid
import json
import os
from pinecone import Pinecone

# 🔐 Variables d'environnement recommandées sur Render !
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("prospectsupport")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/process")
async def process_file(
    data: UploadFile = File(...),
    filename: str = Form(...),
    index_name: str = Form("prospectsupport")
):
    content = await data.read()
    try:
        json_data = json.loads(content)
        text = json_data.get("text", "")
    except Exception as e:
        return {"error": str(e)}
    if not text:
        return {"error": "No text in input"}
    # Découpage simple (1 chunk unique ici, tu peux améliorer si tu veux)
    chunks = [text]
    embeddings = []
    for chunk in chunks:
        response = openai.embeddings.create(input=[chunk], model="text-embedding-3-large")
        embeddings.append(response.data[0].embedding)
    vector_ids = []
    for i, vector in enumerate(embeddings):
        vector_id = str(uuid.uuid4())
        index.upsert([(vector_id, vector, {"source": filename, "chunk": i})])
        vector_ids.append(vector_id)
    return {
        "message": "Success",
        "filename": filename,
        "chunks": len(chunks),
        "vector_ids": vector_ids
    }
