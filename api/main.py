from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens. Alterar para o domínio específico em produção.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Caminhos para o modelo e o vetorizador
MODEL_PATH = "finalized_model.sav"
VECTORIZER_PATH = "count_vectorizer.sav"  # Nome do vetorizador salvo

# Carregando o modelo e o vetorizador
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Modelo e vetorizador carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo ou vetorizador: {e}")
    model = None
    vectorizer = None

# Definição do esquema para entrada
class PredictionRequest(BaseModel):
    text: str  # Entrada é um texto simples para vetorização

@app.get("/")
def root():
    return {"message": "API para predição com CountVectorizer está funcionando!"}

@app.post("/predict")
def predict(data: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Modelo ou vetorizador não está carregado no servidor.")

    try:
        # Vetoriza o texto de entrada
        input_features = vectorizer.transform([data.text])

        # Garante que o formato esteja correto para o modelo
        input_array = input_features.toarray() if hasattr(input_features, "toarray") else input_features

        # Faz a previsão
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao fazer a previsão: {e}")
