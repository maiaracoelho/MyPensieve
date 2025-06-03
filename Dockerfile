FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalações mínimas sem depender do repositório da NVIDIA
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlink para chamar python e pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# Cria diretório de trabalho
WORKDIR /workspace/app

# Copia os arquivos do projeto
COPY . .

# Instala dependências (ajustadas)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python", "src/train.py"]
