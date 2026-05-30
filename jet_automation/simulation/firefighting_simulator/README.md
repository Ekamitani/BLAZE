# Simulador de atuação do jato de água

Este diretório contém o notebook principal do módulo de automação do jato de água para simulação de combate a incêndio.

Notebook principal:

    firefighting_simulator.ipynb

## 1. Objetivo

O objetivo deste notebook é simular, de forma interativa, a atuação de um sistema de direcionamento de jato de água sobre uma região de incêndio sintética.

O notebook permite testar estratégias de trajetória, critérios de priorização de regiões, parâmetros de velocidade, tempo de permanência do cursor e resposta visual da chama ao jato.

Ele pertence ao módulo de automação, não ao módulo de visão computacional.

## 2. Localização no projeto

Este notebook fica em:

    jet_automation/simulation/firefighting_simulator/

A raiz esperada do projeto é a pasta BLAZE.

## 3. Relação com o restante do projeto

O projeto BLAZE está dividido em dois blocos principais:

    vision/
    jet_automation/

O diretório vision/ contém sistemas de visão computacional, como detecção de cilindros e, futuramente, detecção de fogo.

O diretório jet_automation/ contém simulações e futuramente poderá conter lógica relacionada ao controle do direcionamento do jato de água.

Este simulador é usado para estudar a lógica de atuação do jato, independentemente do detector de cilindros.

## 4. Setups paramétricos

O simulador usa setups para salvar e recuperar configurações de simulação, parâmetros da chama, parâmetros do jato e estratégias de trajetória.

Setups oficiais ficam em:

    jet_automation/parameters_setups/official/firefighting_simulator/

Setups locais do usuário ficam em:

    jet_automation/parameters_setups/user/firefighting_simulator/

Arquivo principal:

    setups_simulacao_incendio.json

Os setups locais são ignorados pelo GitHub. Para compartilhar um setup com outras pessoas, ele deve ser promovido para a pasta official/ e enviado com git commit e git push.

Para detalhes sobre esse fluxo, consulte:

    GUIA_SETUPS_PARAMETRICOS.md

## 5. Resultados gerados

Resultados locais do simulador devem ser salvos em:

    jet_automation/results/

Essa pasta é ignorada pelo GitHub para evitar envio de resultados temporários, arquivos gerados e experimentos locais.

## 6. Como executar no VS Code

Abra a pasta BLAZE no VS Code:

    code .

Depois abra o notebook:

    jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb

Selecione o kernel:

    Python (BLAZE)

Execute a célula principal com:

    Shift + Enter

O notebook deve carregar a interface interativa do simulador.

## 7. Problemas comuns

### Erro de widgets no VS Code

Se aparecer erro relacionado a ipywidgets ou jupyter-ipywidget-renderer, atualize o ambiente:

    source .venv/bin/activate
    python -m pip install -U ipywidgets jupyter ipykernel notebook widgetsnbextension jupyterlab_widgets
    python -m ipykernel install --user --name blaze --display-name "Python (BLAZE)"

Depois recarregue o VS Code e selecione novamente o kernel Python (BLAZE).

### Setups não aparecem

Verifique se os setups oficiais existem em:

    jet_automation/parameters_setups/official/firefighting_simulator/

Se o notebook criar setups locais, eles serão salvos em:

    jet_automation/parameters_setups/user/firefighting_simulator/

Essa pasta local não é enviada ao GitHub.

### O Git mostra o notebook modificado após executar

Isso pode acontecer porque o VS Code/Jupyter salva outputs e metadados nos arquivos .ipynb.

Antes de fazer commit, confira:

    git status --short
    git diff --stat

Se a alteração for apenas output/metadado, limpe os outputs antes de enviar ao GitHub.

## 8. Observações

Este notebook não depende do dataset de cilindros.

Ele pode ser executado mesmo que a pasta vision/datasets/ ainda esteja vazia.

Para o detector clássico de cilindros, consulte:

    vision/cylinders_detect/classic_vision/
