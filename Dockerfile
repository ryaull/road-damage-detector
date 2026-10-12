FROM python:3.11-slim

# system libraries OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Docker Spaces as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    PYTHONUNBUFFERED=1
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
