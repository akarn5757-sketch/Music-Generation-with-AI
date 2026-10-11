FROM python:3.10-slim

# System dependencies install karein
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements copy aur install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saare project files copy karein
COPY . .

EXPOSE 7860

# App ko start karein
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]