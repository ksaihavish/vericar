# VeriCar - CPU image for local use and Hugging Face Spaces (Docker SDK).
#   docker build -t vericar .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... vericar      -> http://localhost:7860
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics VERICAR_WORK=/tmp/vericar_work

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN python scripts/download_models.py

RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
