# 1. Base image: Python 3.10.12 slim
FROM python:3.10.12-slim

# 2. Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# 3. Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# 4. Working directory
WORKDIR /app

# 5. Copy & install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
