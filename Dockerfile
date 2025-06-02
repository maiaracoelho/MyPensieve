FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instala Python e dependências
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && apt-get clean

# Symlink para "python"
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Define diretório de trabalho
WORKDIR /workspace/app

# Copia os arquivos do projeto
COPY . .

# Instala pacotes
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando default
CMD ["python", "src/train.py"]
