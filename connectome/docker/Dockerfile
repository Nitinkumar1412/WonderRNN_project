# Replica of the scoring container (docs/submission_guide.md): Python 3.11 slim-bookworm,
# build-essential (gcc for kernel.c), the starter pack's requirements, 1 CPU at run time.
#   docker build -t wnn-scorer -f docker/Dockerfile docker
#   bash tools/docker_check.sh <submission dir>
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip check && pip cache purge

ENV HOME=/app OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
