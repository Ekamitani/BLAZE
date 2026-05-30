# Como contribuir com o projeto BLAZE

Este guia orienta como contribuir com o projeto BLAZE sem comprometer a estabilidade da branch main.

## 1. Antes de começar

Leia primeiro:

    README.md
    GUIA_USO_LOCAL.md
    GUIA_FLUXO_GIT.md

Se a contribuição envolver dados externos, leia também:

    INSTALACAO_DADOS.md

Se a contribuição envolver setups paramétricos, leia:

    GUIA_SETUPS_PARAMETRICOS.md

## 2. Organização das branches

A branch main representa a versão estável do projeto.

A branch dev é usada para desenvolvimento contínuo.

Contribuições maiores devem ser feitas em branches específicas, criadas a partir da dev.

Exemplos:

    yolo-cilindros
    yolo-fogo
    modularizacao-detector-cilindros
    ajuste-setups-parametricos

## 3. Clonar e preparar o projeto

No Linux:

    git clone https://github.com/Ekamitani/BLAZE.git
    cd BLAZE
    bash scripts/setup_linux.sh

No Windows:

    git clone https://github.com/Ekamitani/BLAZE.git
    cd BLAZE
    scripts\setup_windows.bat

Os dados externos devem ser instalados separadamente conforme INSTALACAO_DADOS.md.

## 4. Fluxo recomendado de contribuição

Antes de alterar arquivos:

    git status
    git pull --ff-only

Para mudanças grandes, use a branch dev ou uma branch específica:

    git switch dev
    git pull --ff-only
    git switch -c nome-da-melhoria

Depois de editar e testar:

    git status --short
    git diff --stat
    git add caminho/do/arquivo
    git commit -m "Mensagem objetiva"
    git push

## 5. Pull Requests

Contribuições externas devem ser enviadas por Pull Request.

O fluxo recomendado é:

    branch-de-trabalho -> dev

Depois que a dev estiver estável, as alterações podem ser incorporadas em:

    dev -> main

A branch main deve permanecer funcional e validada.

## 6. Datasets, resultados e arquivos locais

Não envie datasets, resultados, modelos treinados, curadorias locais ou arquivos temporários para o GitHub.

Esses dados devem ficar em pastas ignoradas pelo Git, como:

    vision/datasets/
    vision/results/
    jet_automation/results/
    vision/parameters_setups/user/
    jet_automation/parameters_setups/user/

## 7. Setups paramétricos

Setups locais devem permanecer em user/ enquanto estiverem em teste.

Para comparar setups locais com oficiais:

    python scripts/comparar_setups.py --verbose

Para promover um setup local validado para official/:

    python scripts/promover_setup.py --tipo cilindros_parametricos --id S005

Depois de promover, teste o notebook antes de fazer commit.

## 8. Notebooks

Antes de enviar notebooks ao GitHub, verifique se a alteração é realmente de código ou texto.

Notebooks podem ser marcados como modificados apenas por outputs ou metadados.

Use:

    git status --short
    git diff --stat

Se a mudança for apenas output, limpe os outputs antes do commit.

## 9. Checklist antes de abrir Pull Request

Antes de abrir um PR, confirme:

    - o projeto ainda executa sem erro;
    - os notebooks alterados foram testados;
    - datasets e resultados não foram adicionados ao Git;
    - setups locais não foram enviados por engano;
    - setups oficiais foram testados antes do commit;
    - a mensagem dos commits está clara;
    - o Pull Request descreve o que foi alterado.
