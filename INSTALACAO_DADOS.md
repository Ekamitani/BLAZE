# Instalação dos dados do projeto BLAZE

Este guia explica como instalar os dados externos necessários para executar o notebook de detecção clássica de cilindros.

Os dados não são enviados ao GitHub porque possuem tamanho elevado. Eles devem ser baixados separadamente e colocados na estrutura esperada pelo projeto.

## Pasta no Google Drive

Os arquivos estão disponíveis na pasta:

https://drive.google.com/drive/folders/1EXnf3znUBM4bmq85VITlNfN1ecSvxvb5?usp=drive_link

Baixe os quatro arquivos:

```text
CylinDeRS-1.tar.gz
CylinDeRS-1.sha256
curadoria_cilindros_zoomout.tar.gz
curadoria_cilindros_zoomout.sha256
```

Recomenda-se deixar os arquivos baixados em:

    ~/Downloads/

## Instalação rápida no Linux externo

### 1. Entrar ou clonar o projeto

Se o projeto ainda não existir:

    cd ~/Documents
    git clone https://github.com/Ekamitani/BLAZE.git
    cd BLAZE

Se o projeto já existir:

    cd ~/Documents/BLAZE
    git pull --ff-only

### 2. Instalar o ambiente Python

Dentro da pasta `BLAZE`:

    bash scripts/setup_linux.sh

### 3. Verificar o dataset bruto

Entre na pasta de downloads:

    cd ~/Downloads

Como o arquivo `.sha256` pode conter um caminho antigo, use:

    echo "7ca7fa00b0ef707ade30cf12d3c7d8b3b3e56f6794112acbe46557b29aade919  CylinDeRS-1.tar.gz" | sha256sum -c -

O esperado é:

    CylinDeRS-1.tar.gz: OK

### 4. Extrair o dataset bruto

Entre na pasta do projeto:

    cd ~/Documents/BLAZE

Crie a pasta de destino:

    mkdir -p vision/datasets/cylinders

Extraia o dataset:

    tar -xzf ~/Downloads/CylinDeRS-1.tar.gz -C vision/datasets/cylinders

A pasta final deve ser:

    BLAZE/vision/datasets/cylinders/CylinDeRS-1/

Confira:

    find vision/datasets/cylinders/CylinDeRS-1 -maxdepth 2 -type d | sort

### 5. Verificar a curadoria e o zoom out

Entre na pasta de downloads:

    cd ~/Downloads

Verifique o pacote de curadoria:

    HASH=$(cut -d ' ' -f1 curadoria_cilindros_zoomout.sha256)
    echo "$HASH  curadoria_cilindros_zoomout.tar.gz" | sha256sum -c -

O esperado é:

    curadoria_cilindros_zoomout.tar.gz: OK

### 6. Extrair a curadoria e o zoom out

Entre na pasta do projeto:

    cd ~/Documents/BLAZE

Crie a pasta de resultados:

    mkdir -p vision/results

Extraia a curadoria:

    tar -xzf ~/Downloads/curadoria_cilindros_zoomout.tar.gz -C vision/results

A pasta final deve ser:

    BLAZE/vision/results/classical_cylinder_detector/

Confira os JSONs:

    find vision/results/classical_cylinder_detector -maxdepth 1 -type f -name "*.json" | sort

Confira o dataset derivado com zoom out:

    find vision/results/classical_cylinder_detector/dataset_zoomout_aplicado -maxdepth 2 -type d | sort

### 7. Confirmar que os dados não entram no Git

Dentro de `BLAZE`:

    git status --short

O ideal é não aparecer nada. As pastas `vision/datasets/` e `vision/results/` são ignoradas pelo Git.

### 8. Abrir no VS Code

Dentro da pasta `BLAZE`:

    code .

Se o comando `code` não funcionar, abra o VS Code manualmente e use:

    File -> Open Folder -> BLAZE

### 9. Executar o notebook de cilindros

Abra:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb

Selecione o kernel:

    Python (BLAZE)

Na célula de caminhos, deve aparecer:

    Dataset encontrado.

Na célula de seleção manual, o esperado é algo próximo de:

    treino selecionado: 1977 imagens | JSON: True
    teste selecionado : 310 imagens | JSON: True
    imagens marcadas para zoom out: treino=1279 | teste=221 | percentual=60%
    referências disponíveis: 2287 imagens selecionadas

Se aparecer `JSON: False` ou `zoom out: 0`, a curadoria não foi extraída no local correto.

### 10. Notebook do simulador

O simulador não depende do dataset de cilindros. Para executá-lo, abra:

    jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb

Selecione o kernel `Python (BLAZE)` e execute a célula principal.
