import os
import random
from fastapi import FastAPI, HTTPException
from redis import Redis, ConnectionError as RedisConnectionError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import requests
from pydantic import BaseModel, validator
from typing import List



# Получаем хосты из переменных окружения, чтобы приложение было гибким
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
INFERENCE_SERVICE_URL = os.getenv(
    "INFERENCE_SERVICE_URL", 
    "http://inference-service.default.svc.cluster.local/invocations"
)
app = FastAPI()

class PredictionRequest(BaseModel):
    data: List[List[float]]

    @validator('data', each_item=True)
    def check_features_count(cls, v):
        # Ваша модель была обучена на 10 признаках
        if len(v) != 10:
            raise ValueError('Каждый вектор данных должен содержать ровно 10 признаков')
        return v

# Инициализация клиентов
try:
    redis_client = Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Successfully connected to Redis.")
except RedisConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
    redis_client = None

try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
    # Создаем коллекцию, если ее нет
    try:
         qdrant_client.recreate_collection(
             collection_name="my_collection",
             vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
         )
         print("Qdrant collection 'my_collection' created/recreated.")
    except UnexpectedResponse: # Может быть уже создана другим инстансом
         print("Qdrant collection 'my_collection' already exists.")
    print("Successfully connected to Qdrant.")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

@app.get("/health")
def health_check():
    """Проверяет доступность сервисов."""
    services = {
        "redis": "ok" if redis_client and redis_client.ping() else "error",
        "qdrant": "ok" if qdrant_client else "error", # упрощенная проверка для qdrant
    }
    status_code = 200 if all(s == "ok" for s in services.values()) else 503
    return {"status": "ok" if status_code == 200 else "error", "services": services}

@app.get("/cache-example")
def cache_example():
    """Пример использования кеша Redis."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis is not available")
    
    cached_value = redis_client.get("my_key")
    if cached_value:
        return {"source": "cache", "value": cached_value}

    new_value = f"random_value_{random.randint(1, 100)}"
    redis_client.set("my_key", new_value, ex=10) # Кешируем на 10 секунд
    return {"source": "generated", "value": new_value}

@app.post("/vector-example")
def vector_example():
    """Пример добавления и поиска векторов в Qdrant."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant is not available")

    # Добавляем случайный вектор
    vec_id = random.randint(1, 1000)
    vec = [random.random() for _ in range(4)]
    qdrant_client.upsert(
        collection_name="my_collection",
        points=[models.PointStruct(id=vec_id, vector=vec)],
        wait=True,
    )

    # Ищем ближайший вектор к только что добавленному
    search_result = qdrant_client.search(
        collection_name="my_collection",
        query_vector=vec,
        limit=1
    )
    return {"added_vector": vec, "closest_found": search_result}

@app.post("/predict")
def predict(request_data: PredictionRequest):
    """
    Принимает данные, отправляет их в inference-сервис и возвращает предсказание.
    """
    # 1. Форматируем данные для MLflow-сервера
    columns = [f"feature_{i}" for i in range(10)]
    mlflow_input = {
        "dataframe_split": {
            "columns": columns,
            "data": request_data.data
        }
    }

    try:
        # 2. Отправляем запрос в inference-сервис
        response = requests.post(INFERENCE_SERVICE_URL, json=mlflow_input, timeout=5)
        response.raise_for_status()  # Вызовет ошибку для статусов 4xx/5xx

        # 3. Возвращаем ответ от модели
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Если сервис модели недоступен
        raise HTTPException(
            status_code=503, 
            detail=f"Inference service is unavailable: {e}"
        )
    except Exception as e:
        # Другие возможные ошибки
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {e}"
        )
# Локальное тестирование
# python -m uvicorn main:app --reload
