from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Configuration API KEYS depuis Render (sécurité optimale)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Client OpenAI initialisé de manière retardée
client = None

def get_openai_client():
    global client
    if client is None:
        try:
            # Tentative avec la nouvelle syntaxe OpenAI v1.x
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            print("OPENAI CLIENT INITIALIZED (v1.x)")
        except Exception as e:
            try:
                # Fallback vers l'ancienne syntaxe si nécessaire
                import openai
                openai.api_key = OPENAI_API_KEY
                client = openai
                print("OPENAI CLIENT INITIALIZED (v0.x fallback)")
            except Exception as e2:
                print(f"Erreur initialisation OpenAI: {e}, {e2}")
                raise e
    return client

pc = Pinecone(api_key=PINECONE_API_KEY)

# Configuration Option 1: Maximum Performance
# text-embedding-3-small avec dimensions complètes (1536) sur index prospectsupport1536
EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_DIMENSIONS = 1536
INDEX_NAME = "prospectsupport1536"

index = pc.Index(INDEX_NAME)

def split_text(text, max_chars=15000):
    # Découpe le texte en chunks de 15000 caractères (≈8192 tokens OpenAI)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

@app.post("/process")
async def process_file(
    data: UploadFile = File(...),
    filename: str = Form(...),
    index_name: str = Form("prospectsupport1536")
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
            print("MODEL USED:", EMBEDDING_MODEL)
            print("TARGET DIMENSIONS:", TARGET_DIMENSIONS)
            print("TARGET INDEX:", INDEX_NAME)

            # Utilisation du client avec gestion des deux versions
            openai_client = get_openai_client()

            # Vérification du type de client pour adapter la syntaxe
            if hasattr(openai_client, 'embeddings') and hasattr(openai_client.embeddings, 'create'):
                # Nouvelle syntaxe OpenAI v1.x - dimensions complètes par défaut
                response = openai_client.embeddings.create(
                    input=[chunk],
                    model=EMBEDDING_MODEL
                )
            else:
                # Ancienne syntaxe OpenAI v0.x
                response = openai_client.Embedding.create(
                    input=[chunk],
                    model=EMBEDDING_MODEL
                )

            embedding = response.data[0].embedding
            print("VECTOR LENGTH:", len(embedding))

            # Vérification que les dimensions correspondent
            if len(embedding) != TARGET_DIMENSIONS:
                print(f"ATTENTION: Dimension mismatch! Expected {TARGET_DIMENSIONS}, got {len(embedding)}")
                return {"error": f"Dimension mismatch: Expected {TARGET_DIMENSIONS}, got {len(embedding)}"}

        except Exception as e:
            print("Erreur OpenAI sur le chunk", i, ":", e)
            return {"error": f"Erreur OpenAI sur le chunk {i} : {str(e)}"}

        # Injection dans Pinecone
        try:
            vector_id = str(uuid.uuid4())
            index.upsert([(vector_id, embedding, {"source": filename, "chunk": i})])
            vector_ids.append(vector_id)
            print(f"Chunk {i} indexé avec succès - ID: {vector_id}")
        except Exception as e:
            print("Erreur Pinecone sur le chunk", i, ":", e)
            return {"error": f"Erreur Pinecone sur le chunk {i} : {str(e)}"}

    print("INDEXATION SUCCESS. Fichier:", filename, "Chunks:", len(chunks))
    return {
        "message": "Success",
        "filename": filename,
        "chunks": len(chunks),
        "vector_ids": vector_ids,
        "text_len": len(text),
        "model_used": EMBEDDING_MODEL,
        "dimensions": TARGET_DIMENSIONS,
        "index_used": INDEX_NAME
    }
