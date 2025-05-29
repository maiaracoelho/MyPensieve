FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências básicas
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && apt-get clean

# Symlink para "python"
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copia arquivos
WORKDIR /workspace/app
COPY . .

# Instala pacotes
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando default
CMD ["python", "src/train.py"]
