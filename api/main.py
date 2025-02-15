from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import requests
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs públicas do modelo e vetorizador no Supabase
SUPABASE_MODEL_URL = "https://ltatfnylwvmergpbxshm.supabase.co/storage/v1/object/sign/models/finalized_model.sav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJtb2RlbHMvZmluYWxpemVkX21vZGVsLnNhdiIsImlhdCI6MTczOTYzNDM4OCwiZXhwIjoxNzcxMTcwMzg4fQ.klGjF20T047m9zPk4JG6I1YEEQzFIxQdSZ5hvhDwKQw"
SUPABASE_VECTORIZER_URL = "https://ltatfnylwvmergpbxshm.supabase.co/storage/v1/object/sign/models/count_vectorizer.sav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJtb2RlbHMvY291bnRfdmVjdG9yaXplci5zYXYiLCJpYXQiOjE3Mzk2MzQwNjYsImV4cCI6MTc3MTE3MDA2Nn0.GBZdgmu1Xl5_xmPHCX8r6Nfh7xNJXlytJ8kQRCTeWNU"


def load_model_from_url(url):
    """Baixa o modelo diretamente da URL e carrega na memória sem salvar no disco."""
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        raise Exception(f"Erro ao baixar modelo de {url}: {response.status_code}")

try:
    model = load_model_from_url(SUPABASE_MODEL_URL)
    vectorizer = load_model_from_url(SUPABASE_VECTORIZER_URL)
    print("Modelo e vetorizador carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None
    vectorizer = None

# Definição do esquema para entrada
class PredictionRequest(BaseModel):
    text: str  

@app.get("/")
def root():
    return {"message": "API para predição com CountVectorizer está funcionando!"}

@app.post("/predict")
def predict(data: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Modelo ou vetorizador não está carregado.")

    try:
        input_features = vectorizer.transform([data.text])
        input_array = input_features.toarray() if hasattr(input_features, "toarray") else input_features
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao fazer a previsão: {e}")
