FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://127.0.0.1:5000
ENV MODEL_PATH=cifar10_efficientnetv2.keras

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY streamlit_app.py .
COPY cifar10_efficientnetv2.keras .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000 8501

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 180 app:app & exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
