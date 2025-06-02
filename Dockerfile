# Base CUDA com menos exigências de compatibilidade
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

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
RUN ln -s /usr/bin/python3.8 /usr/bin/python || true

# Define diretório de trabalho
WORKDIR /workspace/app

# Copia os arquivos da aplicação
COPY . .

# Atualiza o pip e instala as dependências
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando padrão para rodar o experimento
CMD ["python", "src/train.py"]
