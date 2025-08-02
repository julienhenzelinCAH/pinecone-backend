from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import uuid
import os
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# 🔐 Clés API depuis les variables d'environnement Render
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("prospectsupport")

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
            return {"error": f"Impossible de décoder le fichier TXT : {str(e)}"}
    elif ext == '.pdf':
        try:
            from io import BytesIO
            import PyPDF2
            pdf = PyPDF2.PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            return {"error": f"Impossible d'extraire le texte PDF : {str(e)}"}
    else:
        return {"error": f"Type de fichier non supporté : {ext}. Utilisez .txt ou .pdf uniquement."}

    if not text.strip():
        return {"error": "Aucun texte extrait du fichier."}

    # Vectorisation OpenAI
    try:
        response = openai.embeddings.create(input=[text], model="text-embedding-3-large")
        embedding = response.data[0].embedding
    except Exception as e:
        return {"error": f"Erreur OpenAI : {str(e)}"}

    # Injection dans Pinecone
    try:
        vector_id = str(uuid.uuid4())
        index.upsert([(vector_id, embedding, {"source": filename})])
    except Exception as e:
        return {"error": f"Erreur Pinecone : {str(e)}"}

    return {
        "message": "Success",
        "filename": filename,
        "vector_id": vector_id,
        "text_len": len(text)
    }
