# inference/Dockerfile

# Используем базовый образ Python
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Устанавливаем mlflow и нужную библиотеку для модели (scikit-learn)
# Также gunicorn для production-ready сервера
RUN pip install --no-cache-dir mlflow[extras] scikit-learn gunicorn pandas
EXPOSE 8080


# Команда для запуска сервера.
# MLFLOW_TRACKING_URI и MODEL_NAME будут переданы как переменные окружения в Kubernetes
CMD mlflow models serve -m "models:/${MODEL_NAME}/Production" -h 0.0.0.0 -p 8080 --no-conda --workers 2

