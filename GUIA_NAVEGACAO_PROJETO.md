# Guia macro de navegação do projeto BLAZE

Este guia indica a ordem recomendada de leitura dos arquivos de documentação do projeto BLAZE.

## 1. Para quem quer apenas usar o projeto

Leia nesta ordem:

    1. README.md
    2. GUIA_USO_LOCAL.md
    3. INSTALACAO_DADOS.md
    4. vision/cylinders_detect/classic_vision/README.md
    5. jet_automation/simulation/firefighting_simulator/README.md

Esse caminho é indicado para quem quer instalar o ambiente, baixar os dados e executar os notebooks.

## 2. Para quem quer entender a organização geral

Leia nesta ordem:

    1. README.md
    2. vision/README.md
    3. jet_automation/README.md
    4. GUIA_SETUPS_PARAMETRICOS.md
    5. GUIA_FLUXO_GIT.md

Esse caminho é indicado para entender a estrutura do repositório, os módulos principais e o fluxo de trabalho.

## 3. Para quem vai colaborar com código

Leia nesta ordem:

    1. README.md
    2. CONTRIBUTING.md
    3. GUIA_FLUXO_GIT.md
    4. GUIA_SETUPS_PARAMETRICOS.md
    5. README específico do módulo que será alterado

Contribuições externas devem ser feitas por branch ou fork e enviadas por Pull Request.

## 4. Para quem vai trabalhar com setups paramétricos

Leia nesta ordem:

    1. GUIA_SETUPS_PARAMETRICOS.md
    2. vision/cylinders_detect/classic_vision/README.md
    3. jet_automation/simulation/firefighting_simulator/README.md

Scripts úteis:

    python scripts/comparar_setups.py --verbose
    python scripts/promover_setup.py --tipo cilindros_parametricos --id S005

## 5. Para o desenvolvimento futuro do YOLO

Antes de iniciar o código YOLO, recomenda-se ler:

    1. README.md
    2. vision/README.md
    3. GUIA_FLUXO_GIT.md
    4. CONTRIBUTING.md

O desenvolvimento do YOLO deve ocorrer preferencialmente na branch dev ou em uma branch específica criada a partir dela.

## 6. Resumo dos documentos principais

    README.md -> visão geral do projeto
    GUIA_USO_LOCAL.md -> instalação local do ambiente
    INSTALACAO_DADOS.md -> instalação de dataset e curadoria
    GUIA_SETUPS_PARAMETRICOS.md -> gerenciamento de setups
    GUIA_FLUXO_GIT.md -> fluxo Git, branches e Pull Requests
    CONTRIBUTING.md -> regras para colaboração
    vision/README.md -> módulo de visão computacional
    jet_automation/README.md -> módulo de automação do jato
