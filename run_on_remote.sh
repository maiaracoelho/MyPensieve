#!/bin/bash

# Define log e tempo
START=$(date +%s)
LOG_FILE=exec_$(date +%Y%m%d_%H%M%S).log

# Executa docker-compose em segundo plano com nohup
nohup /usr/bin/time -v docker-compose --compatibility up > $LOG_FILE 2>&1 &

# Guarda o PID
echo "🟢 Docker rodando em segundo plano com PID $!"
echo "⏱ Logs sendo salvos em $LOG_FILE"
