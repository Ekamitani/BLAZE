# Guia rápido para usar o BLAZE localmente

Este guia explica como rodar os notebooks do projeto BLAZE em um computador local usando Visual Studio Code.

## Opção rápida: instalação automática

No Linux, após clonar o repositório, execute:

~~~bash
bash scripts/setup_linux.sh
~~~

No Windows, após clonar o repositório, execute no Prompt de Comando:

~~~bat
scripts\setup_windows.bat
~~~

Esses scripts criam o ambiente virtual `.venv`, instalam as dependências e registram o kernel Jupyter `Python (BLAZE)`.

## 1. Clonar o repositório

~~~bash
git clone https://github.com/Ekamitani/BLAZE.git
cd BLAZE
~~~

## 2. Criar ambiente virtual

No Linux:

~~~bash
python -m venv .venv
source .venv/bin/activate
~~~

No Windows:

~~~bash
python -m venv .venv
.venv\Scripts\activate
~~~

## 3. Instalar dependências

~~~bash
python -m pip install -U pip
python -m pip install -r requirements.txt
~~~

## 4. Abrir no VS Code

~~~bash
code .
~~~

Instale as extensões recomendadas:

- Python
- Jupyter

Depois abra o notebook desejado e selecione o kernel do ambiente `.venv`.

## 5. Dataset de cilindros

O notebook de cilindros espera o dataset nesta estrutura:

~~~text
vision/datasets/cylinders/CylinDeRS-1/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
~~~

O dataset completo não é enviado ao GitHub. Ele deve ser colocado manualmente nessa pasta.

## 6. Notebooks principais

Detector clássico de cilindros:

~~~text
vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb
~~~

Simulador de atuação do jato de água:

~~~text
jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb
~~~

## 7. Setups

Setups oficiais ficam em:

~~~text
vision/parameters_setups/official/
jet_automation/parameters_setups/official/
~~~

Ao executar os notebooks, os setups oficiais são copiados automaticamente para uma pasta local de usuário:

~~~text
vision/parameters_setups/user/
jet_automation/parameters_setups/user/
~~~

A pasta `user/` não é enviada ao GitHub.

## 8. Resultados

Resultados gerados pelos notebooks ficam em:

~~~text
vision/results/
jet_automation/results/
~~~

Essas pastas também não são enviadas ao GitHub.
