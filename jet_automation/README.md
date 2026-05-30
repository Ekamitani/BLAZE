# Módulo de automação do jato

Esta pasta concentra os notebooks, parâmetros, resultados e futuras rotinas relacionadas à automação do direcionamento do jato de água no projeto BLAZE.

## 1. Organização geral

A estrutura principal deste módulo é:

    jet_automation/
    ├── simulation/
    ├── parameters_setups/
    └── results/

## 2. Objetivo do módulo

O objetivo deste módulo é estudar e desenvolver estratégias de atuação do jato de água, incluindo simulações de trajetória, priorização de regiões, resposta da chama ao jato e parâmetros de controle.

Este módulo é separado do módulo de visão computacional. Enquanto vision/ trata da percepção visual, jet_automation/ trata da lógica de atuação e simulação do jato.

## 3. Simulador disponível

O simulador atual fica em:

    jet_automation/simulation/firefighting_simulator/

Notebook principal:

    jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb

Esse notebook permite simular a atuação de um cursor/jato sobre uma região de incêndio sintética, testando estratégias de trajetória, velocidade, permanência sobre focos e priorização de regiões.

Para detalhes específicos, consulte:

    jet_automation/simulation/firefighting_simulator/README.md

## 4. Setups paramétricos

Os setups oficiais do módulo de automação ficam em:

    jet_automation/parameters_setups/official/

Os setups locais do usuário ficam em:

    jet_automation/parameters_setups/user/

Para o simulador de incêndio, o arquivo oficial principal é:

    jet_automation/parameters_setups/official/firefighting_simulator/setups_simulacao_incendio.json

Os setups locais são ignorados pelo GitHub. Para compartilhar um setup, ele deve ser promovido para a pasta official/.

Para comparar e promover setups, use os scripts:

    python scripts/comparar_setups.py --verbose
    python scripts/promover_setup.py --tipo simulador_incendio --id CENARIO_001

Para detalhes completos, consulte:

    GUIA_SETUPS_PARAMETRICOS.md

## 5. Resultados

Resultados gerados pelos notebooks de automação devem ficar em:

    jet_automation/results/

Essa pasta é ignorada pelo GitHub, exceto pelo arquivo .gitkeep.

Isso evita enviar resultados temporários, arquivos gerados, saídas intermediárias e experimentos locais para o repositório.

## 6. Relação com o módulo de visão

O módulo vision/ é responsável por detectar ou estimar informações a partir de imagens.

O módulo jet_automation/ é responsável por simular e futuramente controlar a atuação do jato.

No futuro, a saída de modelos de visão computacional poderá alimentar o módulo de automação, por exemplo indicando regiões prioritárias, focos de incêndio, obstáculos ou elementos de risco.

## 7. Fluxo recomendado de uso

Para usar este módulo em outro computador:

    1. Clonar ou atualizar o repositório BLAZE.
    2. Instalar o ambiente Python com scripts/setup_linux.sh ou scripts/setup_windows.bat.
    3. Abrir o projeto inteiro no VS Code.
    4. Selecionar o kernel Python (BLAZE).
    5. Executar o notebook do simulador.

Notebook principal:

    jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb

Este notebook não depende do dataset de cilindros.

## 8. Cuidados ao versionar alterações

Antes de começar a editar:

    git pull --ff-only

Depois de editar ou testar notebooks:

    git status --short
    git diff --stat

Evite enviar resultados, outputs temporários e setups locais para o GitHub.

Os diretórios de resultados e setups locais são protegidos pelo .gitignore, mas é sempre recomendado conferir o git status antes de commitar.
