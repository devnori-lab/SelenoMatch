# Free live demo on Hugging Face Spaces (Docker SDK, CPU). Also works anywhere Docker runs.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV TORCH_HOME=/app/.torch PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt
# bake the pretrained LoFTR weights into the image so first request doesn't download
RUN python -c "import kornia.feature as KF; KF.LoFTR(pretrained='outdoor')"
COPY . .
RUN chmod -R 777 /app
WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
