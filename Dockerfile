FROM tensorflow/tensorflow:2.4.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && apt-get clean

# Define diretório de trabalho
WORKDIR /workspace/app

# Copia os arquivos do projeto
COPY . .

# Instala dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando padrão
CMD ["python", "src/train.py"]
