FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libffi-dev libssl-dev python3-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY subset_data ./subset_data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
