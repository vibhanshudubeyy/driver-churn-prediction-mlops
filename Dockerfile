FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn pandas scikit-learn
EXPOSE 8000
CMD ["uvicorn", "deployment.api:app", "--host", "127.0.0.1", "--port", "8080"]