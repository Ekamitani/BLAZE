# Módulo de visão computacional

Esta pasta concentra os notebooks, datasets, parâmetros e resultados relacionados às aplicações de visão computacional do projeto BLAZE.

## 1. Organização geral

A estrutura principal deste módulo é:

    vision/
    ├── cylinders_detect/
    ├── fire_detect/
    ├── datasets/
    ├── parameters_setups/
    └── results/

## 2. Detectores disponíveis

### Detector clássico de cilindros

O detector clássico de cilindros está em:

    vision/cylinders_detect/classic_vision/

Notebook principal:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb

Esse notebook combina técnicas clássicas de visão computacional, como pré-processamento, Canny, Hough, análise geométrica, HOG e SVM.

Para detalhes específicos, consulte:

    vision/cylinders_detect/classic_vision/README.md

### Detector de fogo

A estrutura para detecção de fogo está reservada em:

    vision/fire_detect/

Essa parte ainda pode receber notebooks clássicos, modelos baseados em aprendizado de máquina ou modelos YOLO em versões futuras.

## 3. Datasets

Os datasets usados pelos notebooks de visão computacional devem ficar em:

    vision/datasets/

O dataset de cilindros deve ser instalado em:

    vision/datasets/cylinders/CylinDeRS-1/

Datasets completos não são enviados ao GitHub, pois podem ser grandes.

Para instalar o dataset de cilindros e a curadoria manual, consulte:

    INSTALACAO_DADOS.md

A pasta vision/datasets/ possui um README específico com a estrutura esperada dos dados.

## 4. Setups paramétricos

Os setups oficiais de visão computacional ficam em:

    vision/parameters_setups/official/

Os setups locais do usuário ficam em:

    vision/parameters_setups/user/

Os setups locais são ignorados pelo GitHub. Para compartilhar um setup, ele deve ser promovido para a pasta official/.

Para comparar e promover setups, use os scripts:

    python scripts/comparar_setups.py --verbose
    python scripts/promover_setup.py --tipo cilindros_parametricos --id S005

Para detalhes completos, consulte:

    GUIA_SETUPS_PARAMETRICOS.md

## 5. Resultados

Resultados gerados pelos notebooks de visão computacional devem ficar em:

    vision/results/

Essa pasta é ignorada pelo GitHub, exceto pelo arquivo .gitkeep.

Isso evita enviar modelos treinados, imagens processadas, curadorias locais, datasets derivados e arquivos temporários para o repositório.

## 6. Fluxo recomendado de uso

Para usar este módulo em outro computador:

    1. Clonar o repositório BLAZE.
    2. Instalar o ambiente Python com scripts/setup_linux.sh ou scripts/setup_windows.bat.
    3. Instalar os datasets externos conforme INSTALACAO_DADOS.md.
    4. Abrir o projeto inteiro no VS Code.
    5. Selecionar o kernel Python (BLAZE).
    6. Executar o notebook desejado.

Para o detector clássico de cilindros, o notebook principal é:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb

## 7. Cuidados ao versionar alterações

Antes de começar a editar:

    git pull --ff-only

Depois de editar ou testar notebooks:

    git status --short
    git diff --stat

Evite enviar datasets, resultados, outputs temporários e setups locais para o GitHub.

Os diretórios de dados e resultados são protegidos pelo .gitignore, mas é sempre recomendado conferir o git status antes de commitar.

## 8. Relação com o módulo de automação

O módulo vision/ é responsável pelas etapas de percepção visual.

O módulo jet_automation/ é responsável pelas simulações e futuras estratégias de controle do jato.

No futuro, os resultados de visão computacional poderão alimentar decisões do módulo de automação.
