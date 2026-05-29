# Datasets

Esta pasta é reservada para datasets utilizados pelos notebooks de visão computacional.

Por padrão, os datasets completos não são enviados ao GitHub, pois podem ser grandes.

## Dataset de cilindros

O notebook de detecção clássica de cilindros espera a seguinte estrutura local:

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

No servidor Jupyter da UFSC, pode-se usar um link simbólico para evitar duplicar arquivos:

~~~bash
ln -s /caminho/para/CylinDeRS-1 vision/datasets/cylinders/CylinDeRS-1
~~~

Em outro computador, a pessoa pode copiar ou baixar o dataset e colocá-lo manualmente nessa pasta.

## Observação

A pasta `vision/datasets/` está protegida pelo `.gitignore`, portanto datasets locais não serão enviados ao GitHub por acidente.
