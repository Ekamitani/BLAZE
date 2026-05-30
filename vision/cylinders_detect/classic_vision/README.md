# Detector clássico de cilindros industriais

Este diretório contém o notebook principal do sistema de visão computacional clássico para detecção de cilindros industriais.

Notebook principal:

    detector_cilindros_hough_hog_svm_interativo.ipynb

## 1. Objetivo

O objetivo deste notebook é detectar cilindros industriais em imagens usando técnicas clássicas de visão computacional, combinando pré-processamento, detecção de bordas, Transformada de Hough, análise geométrica, HOG e SVM.

O notebook foi desenvolvido para permitir ajuste interativo de parâmetros, curadoria manual de imagens, aplicação de zoom out em imagens selecionadas, treinamento de modelo e validação visual dos resultados.

## 2. Localização no projeto

Este notebook fica em:

    vision/cylinders_detect/classic_vision/

A raiz esperada do projeto é a pasta BLAZE.

## 3. Dados necessários

O notebook utiliza o dataset CylinDeRS-1, que não é enviado ao GitHub por causa do tamanho.

O dataset deve ser instalado em:

    vision/datasets/cylinders/CylinDeRS-1/

A estrutura esperada é:

    CylinDeRS-1/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Para instalar o dataset e a curadoria manual, consulte o arquivo:

    INSTALACAO_DADOS.md

## 4. Curadoria manual e zoom out

O notebook permite selecionar manualmente quais imagens serão usadas nas etapas de treino, teste e validação visual.

Além disso, algumas imagens podem ser marcadas para receber zoom out artificial, gerando versões derivadas para melhorar a avaliação do detector em diferentes escalas.

Esses arquivos são locais e ficam em:

    vision/results/classical_cylinder_detector/

Arquivos importantes:

    imagens_selecionadas_treino.json
    imagens_selecionadas_teste.json
    imagens_zoomout_treino.json
    imagens_zoomout_teste.json
    zoomout_config.json
    dataset_zoomout_manifest.json
    dataset_zoomout_aplicado/

Esses arquivos não são enviados ao GitHub, pois ficam dentro de vision/results/.

Para compartilhar a curadoria com outra pessoa, use o pacote externo descrito em:

    INSTALACAO_DADOS.md

## 5. Setups paramétricos

O notebook usa setups paramétricos para salvar e recuperar configurações de pré-processamento, detecção, refinamento e expansão.

Setups oficiais ficam em:

    vision/parameters_setups/official/cylinders_detect/

Setups locais do usuário ficam em:

    vision/parameters_setups/user/cylinders_detect/

Arquivos principais:

    setups_parametricos.json
    setups_refinamento_expansao.json

Para aprender a promover setups locais para setups oficiais compartilhados pelo GitHub, consulte:

    GUIA_SETUPS_PARAMETRICOS.md

## 6. Resultados gerados

Os resultados do notebook são salvos em:

    vision/results/classical_cylinder_detector/
    vision/results/geometric_realtime_detector/

Essas pastas são ignoradas pelo GitHub. Isso evita enviar modelos, imagens processadas, curadorias locais e arquivos temporários para o repositório.

## 7. Como executar no VS Code

Abra a pasta BLAZE no VS Code:

    code .

Depois abra o notebook:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb

Selecione o kernel:

    Python (BLAZE)

Execute as células com:

    Shift + Enter

Na célula inicial de caminhos, o esperado é aparecer:

    Dataset encontrado.

Na célula de seleção manual, se a curadoria tiver sido instalada corretamente, o esperado é algo próximo de:

    treino selecionado: 1977 imagens | JSON: True
    teste selecionado : 310 imagens | JSON: True
    imagens marcadas para zoom out: treino=1279 | teste=221 | percentual=60%
    referências disponíveis: 2287 imagens selecionadas

## 8. Problemas comuns

### Dataset não encontrado

Verifique se a pasta existe exatamente em:

    vision/datasets/cylinders/CylinDeRS-1/

Se ela estiver em outro lugar, extraia novamente o pacote conforme INSTALACAO_DADOS.md.

### JSON: False ou zoom out igual a zero

Isso indica que a curadoria manual não foi encontrada.

Verifique se a pasta existe:

    vision/results/classical_cylinder_detector/

e se os arquivos JSON de seleção e zoom out estão presentes.

### Erro de widgets no VS Code

Se aparecer erro relacionado a ipywidgets ou jupyter-ipywidget-renderer, atualize o ambiente:

    source .venv/bin/activate
    python -m pip install -U ipywidgets jupyter ipykernel notebook widgetsnbextension jupyterlab_widgets
    python -m ipykernel install --user --name blaze --display-name "Python (BLAZE)"

Depois recarregue o VS Code e selecione novamente o kernel Python (BLAZE).

### O Git mostra arquivos modificados após rodar o notebook

Isso pode acontecer porque o VS Code/Jupyter salva outputs e metadados nos arquivos .ipynb.

Antes de fazer commit, confira:

    git status --short
    git diff --stat

Se a alteração for apenas output/metadado, limpe os outputs antes de enviar ao GitHub.

## 9. Observações

Este notebook faz parte do módulo de visão computacional do BLAZE.

Para o simulador de atuação do jato de água, consulte:

    jet_automation/simulation/firefighting_simulator/
