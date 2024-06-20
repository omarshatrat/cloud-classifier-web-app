FROM --platform=linux/x86_64 python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD ["streamlit", "run", "--server.port=80",  "app.py"]