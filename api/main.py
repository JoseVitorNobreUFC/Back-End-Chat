from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens. Alterar para domínios específicos em produção.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs públicas dos arquivos no Supabase
SUPABASE_MODEL_URL = "https://ltatfnylwvmergpbxshm.supabase.co/storage/v1/object/sign/models/finalized_model.sav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJtb2RlbHMvZmluYWxpemVkX21vZGVsLnNhdiIsImlhdCI6MTczOTYzNDM4OCwiZXhwIjoxNzcxMTcwMzg4fQ.klGjF20T047m9zPk4JG6I1YEEQzFIxQdSZ5hvhDwKQw"
SUPABASE_VECTORIZER_URL = "https://ltatfnylwvmergpbxshm.supabase.co/storage/v1/object/sign/models/count_vectorizer.sav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJtb2RlbHMvY291bnRfdmVjdG9yaXplci5zYXYiLCJpYXQiOjE3Mzk2MzQwNjYsImV4cCI6MTc3MTE3MDA2Nn0.GBZdgmu1Xl5_xmPHCX8r6Nfh7xNJXlytJ8kQRCTeWNU"

# Caminhos locais onde os arquivos serão armazenados
LOCAL_MODEL_PATH = "finalized_model.sav"
LOCAL_VECTORIZER_PATH = "count_vectorizer.sav"

def download_file(url, local_path):
    """Baixa um arquivo do Supabase se ele não existir."""
    if not os.path.exists(local_path):
        print(f"Baixando {local_path} do Supabase...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"{local_path} baixado com sucesso.")
        else:
            print(f"Erro ao baixar {local_path}: {response.status_code}")

# Baixar os arquivos se necessário
download_file(SUPABASE_MODEL_URL, LOCAL_MODEL_PATH)
download_file(SUPABASE_VECTORIZER_URL, LOCAL_VECTORIZER_PATH)

# Carregar modelo e vetorizador
try:
    model = joblib.load(LOCAL_MODEL_PATH)
    vectorizer = joblib.load(LOCAL_VECTORIZER_PATH)
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

@app.get("/debug-files")
def debug_files():
    """Verifica se os arquivos foram baixados corretamente."""
    model_exists = os.path.exists(LOCAL_MODEL_PATH)
    vectorizer_exists = os.path.exists(LOCAL_VECTORIZER_PATH)

    # Obtém o diretório atual e lista os arquivos
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)

    return {
        "model_path": LOCAL_MODEL_PATH,
        "vectorizer_path": LOCAL_VECTORIZER_PATH,
        "model_exists": model_exists,
        "vectorizer_exists": vectorizer_exists,
        "current_directory": current_dir,
        "files_in_directory": files_in_dir
    }

@app.get("/test-load")
def test_load():
    """Verifica se o modelo e vetorizador são carregados corretamente."""
    try:
        model_test = joblib.load(LOCAL_MODEL_PATH)
        vectorizer_test = joblib.load(LOCAL_VECTORIZER_PATH)
        return {"message": "Modelo e vetorizador carregados com sucesso!"}
    except Exception as e:
        return {"error": str(e)}
