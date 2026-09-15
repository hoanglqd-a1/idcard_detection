FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_AUTOINSTALL=false PIP_DEFAULT_TIMEOUT=180
WORKDIR /app

# CPU wheels avoid shipping CUDA libraries in the API image.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/constraints.txt /tmp/dependencies/
# These official CPython 3.12 / Linux amd64 CPU wheels avoid the index's
# redirected download host. Keep versions aligned with constraints.txt.
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl
ARG TORCHVISION_WHEEL_URL=https://download.pytorch.org/whl/cpu/torchvision-0.29.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl
RUN pip install --no-cache-dir --no-deps -c /tmp/dependencies/constraints.txt "$TORCH_WHEEL_URL" "$TORCHVISION_WHEEL_URL" \
    && pip install --no-cache-dir -c /tmp/dependencies/constraints.txt -r /tmp/dependencies/requirements.txt

COPY backend/app backend/app
COPY training/detection.py training/detection.py
COPY training/utils training/utils
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
