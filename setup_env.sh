#!/bin/bash

# ⚙️ CONFIGURAÇÕES INICIAIS
PYTHON_VERSION="3.10.13"
VENV_NAME="mypensieve-env"

echo "📦 Atualizando pacotes do sistema..."
sudo apt-get update -y && sudo apt-get install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
  liblzma-dev python-openssl git

# 🔧 INSTALAÇÃO DO PYENV
if [ ! -d "$HOME/.pyenv" ]; then
  echo "🔧 Instalando pyenv..."
  curl https://pyenv.run | bash
fi

# 🔁 CONFIGURAÇÃO DO SHELL (idempotente)
SHELL_CONFIG="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
  SHELL_CONFIG="$HOME/.zshrc"
fi

echo "🧠 Configurando pyenv no $SHELL_CONFIG..."

grep -qxF 'export PATH="$HOME/.pyenv/bin:$PATH"' $SHELL_CONFIG || echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $SHELL_CONFIG
grep -qxF 'eval "$(pyenv init --path)"' $SHELL_CONFIG || echo 'eval "$(pyenv init --path)"' >> $SHELL_CONFIG
grep -qxF 'eval "$(pyenv init -)"' $SHELL_CONFIG || echo 'eval "$(pyenv init -)"' >> $SHELL_CONFIG
grep -qxF 'eval "$(pyenv virtualenv-init -)"' $SHELL_CONFIG || echo 'eval "$(pyenv virtualenv-init -)"' >> $SHELL_CONFIG

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 🐍 INSTALAÇÃO DO PYTHON
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
  echo "🐍 Instalando Python $PYTHON_VERSION via pyenv..."
  pyenv install $PYTHON_VERSION
fi

# 🌱 CRIAÇÃO DO AMBIENTE VIRTUAL
if ! pyenv virtualenvs --bare | grep -qx "$VENV_NAME"; then
  echo "🌱 Criando ambiente virtual $VENV_NAME..."
  pyenv virtualenv $PYTHON_VERSION $VENV_NAME
fi

# ✅ ATIVA O AMBIENTE
echo "✅ Ativando ambiente $VENV_NAME..."
pyenv activate $VENV_NAME

# 🚀 INSTALA DEPENDÊNCIAS DO PROJETO
REQ_PATH="./requirements.txt"
if [ -f "$REQ_PATH" ]; then
  echo "📦 Instalando dependências do requirements.txt..."
  pip install --upgrade pip
  pip install -r "$REQ_PATH"
else
  echo "⚠️ requirements.txt não encontrado em $PWD"
fi

echo
