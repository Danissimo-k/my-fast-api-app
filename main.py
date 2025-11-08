import os
import random
from fastapi import FastAPI, HTTPException
from redis import Redis, ConnectionError as RedisConnectionError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Получаем хосты из переменных окружения, чтобы приложение было гибким
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

app = FastAPI()

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

# Локальное тестирование
# python -m uvicorn main:app --reload
