@echo off
echo === BLAZE: configuracao automatica para Windows ===

echo.
echo 1) Verificando Python...
python --version

echo.
echo 2) Criando ambiente virtual .venv...
python -m venv .venv

echo.
echo 3) Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo.
echo 4) Atualizando pip...
python -m pip install -U pip

echo.
echo 5) Instalando dependencias...
python -m pip install -r requirements.txt

echo.
echo 6) Registrando kernel Jupyter...
python -m ipykernel install --user --name blaze --display-name "Python (BLAZE)"

echo.
echo Configuracao concluida.
echo.
echo Para usar:
echo .venv\Scripts\activate
echo code .
echo.
echo No VS Code, selecione o kernel: Python (BLAZE)

pause
