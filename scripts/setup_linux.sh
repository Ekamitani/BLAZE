#!/usr/bin/env bash

set -e

echo "=== BLAZE: configuração automática para Linux ==="

echo ""
echo "1) Verificando Python..."
python3 --version

echo ""
echo "2) Criando ambiente virtual .venv..."
python3 -m venv .venv

echo ""
echo "3) Ativando ambiente virtual..."
source .venv/bin/activate

echo ""
echo "4) Atualizando pip..."
python -m pip install -U pip

echo ""
echo "5) Instalando dependências..."
python -m pip install -r requirements.txt

echo ""
echo "6) Registrando kernel Jupyter..."
python -m ipykernel install --user --name blaze --display-name "Python (BLAZE)"

echo ""
echo "Configuração concluída."
echo ""
echo "Para usar:"
echo "source .venv/bin/activate"
echo "code ."
echo ""
echo "No VS Code, selecione o kernel: Python (BLAZE)"
