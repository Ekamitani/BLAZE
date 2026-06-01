# Módulo YOLO — Detecção de cilindros industriais

Este módulo contém o notebook YOLO para detecção de cilindros industriais no projeto BLAZE.

Notebook principal:

`vision/cylinders_detect/YOLO/detector_cilindros_yolo_interativo.ipynb`

## Uso recomendado

O notebook pode ser usado em dois ambientes:

1. **Jupyter da UFSC**
   - recomendado para treinamento com GPU compartilhada;
   - permite seleção manual da GPU;
   - normalmente não acessa a webcam local do usuário.

2. **VS Code local**
   - recomendado para validação, inferência, vídeo salvo e webcam local;
   - pode usar CPU ou GPU local, se disponível.

## Dataset

O módulo YOLO usa o mesmo dataset externo do detector clássico:

`vision/datasets/cylinders/CylinDeRS-1/`

Esse dataset não é versionado no GitHub. A instalação deve seguir:

`INSTALACAO_DADOS.md`

## Peso oficial treinado

Não é necessário treinar novamente para usar o detector YOLO.

No notebook, use:

`Passo 19 → Usar peso oficial da Release`

O peso é baixado automaticamente da GitHub Release:

`v0.2.0-yolo-cylinders`

e salvo em:

`vision/results/yolo_cylinder_detector/weights/yolo_cylinders_best.pt`

A integridade é verificada com:

`yolo_cylinders_best.sha256`

## Vídeo salvo — Passo 28

O Passo 28 executa inferência contínua em vídeo salvo dentro do notebook.

Coloque vídeos de teste em:

`vision/datasets/cylinders/videos/`

Exemplo:

`mkdir -p vision/datasets/cylinders/videos`

`cp /caminho/para/video.webm vision/datasets/cylinders/videos/`

Depois, no notebook, clique em:

`Passo 28 → Atualizar lista`

## Webcam local — Passo 29

O Passo 29 usa a webcam local dentro do notebook, sem `cv2.imshow()`.

Esse passo é recomendado para:

`VS Code local`

Não é recomendado para:

`Jupyter remoto da UFSC`

porque o servidor remoto normalmente não acessa a webcam do computador do usuário.

## Configuração da webcam no Linux

Verifique se existem dispositivos de vídeo:

`ls -l /dev/video*`

Verifique se seu usuário pertence ao grupo `video`:

`groups`

Se `video` não aparecer, adicione seu usuário ao grupo:

`sudo usermod -aG video $USER`

Depois reinicie a sessão ou o computador:

`sudo reboot`

Após reiniciar, confira novamente:

`groups`

## Teste rápido da webcam com OpenCV

Na raiz do BLAZE, com o ambiente Python ativo, rode:

`timeout 8s python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print('Abriu?', cap.isOpened()); ret, frame = cap.read() if cap.isOpened() else (False, None); print('Frame capturado?', ret); print('Tamanho:', frame.shape if frame is not None else None); cap.release()"`

Resultado esperado:

`Abriu? True`

`Frame capturado? True`

Nesse caso, use no notebook:

`Webcam index: 0`

## Saídas geradas

Resultados, pesos, vídeos processados e arquivos de treino são salvos em:

`vision/results/yolo_cylinder_detector/`

Essa pasta é ignorada pelo Git.