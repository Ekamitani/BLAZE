# Detector clássico de cilindros — treino/configuração e webcam

Este módulo contém duas formas de uso do detector clássico de cilindros por Hough, arcos, HOG e SVM.

## 1. Notebook de treino e ajuste paramétrico

Arquivo:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo.ipynb

Use este notebook no Jupyter para:

- ajustar parâmetros de pré-processamento, Hough, pares de retas, arcos e BBoxes;
- salvar setups paramétricos em JSON;
- treinar o classificador HOG/SVM;
- validar visualmente o detector.

Setups versionados no GitHub:

    vision/parameters_setups/user/cylinders_detect/setups_parametricos_user.json
    vision/parameters_setups/user/cylinders_detect/setups_refinamento_expansao_user.json

Modelo treinado salvo localmente:

    vision/results/classical_cylinder_detector/modelo_hog_svm_cilindros.joblib
    vision/results/classical_cylinder_detector/scaler_hog_svm_cilindros.joblib
    vision/results/classical_cylinder_detector/metadata_modelo.json

Os arquivos de modelo e resultados não são versionados no GitHub.

## 2. Aplicação em webcam no VS Code

Arquivo:

    vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_webcam_runtime.py

Este script executa somente inferência em tempo real. Ele não treina o modelo.

Ele:

- carrega os setups JSON salvos no projeto;
- carrega o modelo HOG/SVM treinado localmente;
- abre a webcam com OpenCV;
- mostra somente a BBox final da detecção.

## Comandos úteis

Listar setups disponíveis:

    python vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_webcam_runtime.py --list-setups

Rodar com um setup específico:

    python vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_webcam_runtime.py --setup NOME_DO_SETUP

Rodar usando parâmetros registrados no último treino, quando disponíveis no metadata local:

    python vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_webcam_runtime.py --trained-params

Rodar com a webcam padrão:

    python vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_webcam_runtime.py

Sair da janela da webcam:

    pressione q

## Observação importante

Para um usuário externo usar a webcam, é necessário ter localmente um modelo treinado em:

    vision/results/classical_cylinder_detector/modelo_hog_svm_cilindros.joblib

Caso esse arquivo não exista, primeiro execute o notebook de treino/configuração no Jupyter.
