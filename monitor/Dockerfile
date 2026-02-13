FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

# Run unbuffered to ensure logs show up immediately
CMD ["python", "-u", "agent.py"]
