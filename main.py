from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI  # Import corrigé pour v1.x
import uuid
import os
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Configuration API KEYS depuis Render (sécurité optimale)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialisation du client OpenAI (syntaxe v1.x)
client = OpenAI(api_key=OPENAI_API_KEY)

print("OPENAI CLIENT INITIALIZED")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("prospectsupport")

def split_text(text, max_chars=15000):
    # Découpe le texte en chunks de 15000 caractères (≈8192 tokens OpenAI)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

@app.post("/process")
async def process_file(
    data: UploadFile = File(...),
    filename: str = Form(...),
    index_name: str = Form("prospectsupport")
):
    ext = os.path.splitext(filename)[1].lower()
    content = await data.read()
    text = ""

    # Extraction du texte selon le type de fichier
    if ext == '.txt':
        try:
            text = content.decode('utf-8')
        except Exception as e:
            print("Erreur TXT:", e)
            return {"error": f"Impossible de décoder le fichier TXT : {str(e)}"}
    elif ext == '.pdf':
        try:
            from io import BytesIO
            import PyPDF2
            pdf = PyPDF2.PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print("Erreur PDF:", e)
            return {"error": f"Impossible d'extraire le texte PDF : {str(e)}"}
    else:
        print("Type de fichier non supporté :", ext)
        return {"error": f"Type de fichier non supporté : {ext}. Utilisez .txt ou .pdf uniquement."}

    if not text.strip():
        print("Aucun texte extrait du fichier.")
        return {"error": "Aucun texte extrait du fichier."}

    # Découpage en chunks pour OpenAI
    chunks = split_text(text)
    vector_ids = []

    for i, chunk in enumerate(chunks):
        try:
            print("MODEL USED:", "text-embedding-3-small")
            # Syntaxe corrigée pour OpenAI v1.x
            response = client.embeddings.create(
                input=[chunk],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            print("VECTOR LENGTH:", len(embedding))
        except Exception as e:
            print("Erreur OpenAI sur le chunk", i, ":", e)
            return {"error": f"Erreur OpenAI sur le chunk {i} : {str(e)}"}

        # Injection dans Pinecone
        try:
            vector_id = str(uuid.uuid4())
            index.upsert([(vector_id, embedding, {"source": filename, "chunk": i})])
            vector_ids.append(vector_id)
        except Exception as e:
            print("Erreur Pinecone sur le chunk", i, ":", e)
            return {"error": f"Erreur Pinecone sur le chunk {i} : {str(e)}"}

    print("INDEXATION SUCCESS. Fichier:", filename, "Chunks:", len(chunks))
    return {
        "message": "Success",
        "filename": filename,
        "chunks": len(chunks),
        "vector_ids": vector_ids,
        "text_len": len(text)
    }
