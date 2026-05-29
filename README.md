# BLAZE

Projeto para desenvolvimento de sistemas de visão computacional e automação aplicados a robôs de combate a incêndio.

## Estrutura do projeto

~~~text
BLAZE/
├── vision/
│   ├── cylinders_detect/
│   │   ├── classic_vision/
│   │   └── YOLO/
│   ├── fire_detect/
│   │   ├── classic_vision/
│   │   └── YOLO/
│   ├── datasets/
│   ├── parameters_setups/
│   └── results/
│
├── jet_automation/
│   ├── simulation/
│   ├── parameters_setups/
│   └── results/
│
├── src/
├── requirements.txt
└── README.md
~~~

## Instalação recomendada

Clone o repositório:

~~~bash
git clone https://github.com/Ekamitani/BLAZE.git
cd BLAZE
~~~

Crie um ambiente virtual:

~~~bash
python -m venv .venv
~~~

Ative o ambiente no Linux:

~~~bash
source .venv/bin/activate
~~~

Ative o ambiente no Windows:

~~~bash
.venv\Scripts\activate
~~~

Instale as dependências:

~~~bash
python -m pip install -U pip
python -m pip install -r requirements.txt
~~~

## Uso no VS Code

1. Abra a pasta `BLAZE` no Visual Studio Code.
2. Instale as extensões Python e Jupyter.
3. Abra o notebook desejado.
4. Selecione o kernel Python do ambiente `.venv`.

## Organização dos setups

Os setups oficiais ficam em:

~~~text
vision/parameters_setups/official/
jet_automation/parameters_setups/official/
~~~

Os setups criados localmente pelo usuário ficam em:

~~~text
vision/parameters_setups/user/
jet_automation/parameters_setups/user/
~~~

As pastas `user/` não são enviadas ao GitHub, pois são ignoradas pelo `.gitignore`.

## Organização dos resultados

Resultados gerados pelos notebooks devem ser salvos em:

~~~text
vision/results/
jet_automation/results/
~~~

Essas pastas também não são enviadas ao GitHub, exceto pelo arquivo `.gitkeep`.
