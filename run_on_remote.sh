#!/bin/bash

# Caminho do projeto
PROJECT_DIR=MyPensieve

# Entrar na pasta do projeto
cd "$PROJECT_DIR" || { echo "❌ Pasta $PROJECT_DIR não encontrada"; exit 1; }

# Garantir que logs e test_results existem
mkdir -p src/logs src/test_results

# Subir o container em modo detached (background)
echo "🚀 Iniciando container com Docker Compose..."
docker-compose up -d --build

# Aguardar alguns segundos para inicializar
sleep 5

# Verificar se o container está rodando
echo "🔍 Verificando status do container..."
docker-compose ps

# Mostrar logs em tempo real
echo "📜 Logs em tempo real (Ctrl+C para sair):"
docker-compose logs -f
