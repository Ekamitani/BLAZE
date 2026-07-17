# BLAZE

Projeto universitário de desenvolvimento de sistemas de visão computacional, sensoriamento e automação aplicados a um robô de apoio ao combate a incêndios.

A versão atual inclui:

- detecção de cilindros industriais por visão clássica com HOG/SVM;
- detecção de cilindros industriais com YOLO;
- monitoramento monocular de fogo em imagens RGB;
- visão térmica estéreo aplicada à automação do jato de água;
- simulador de planejamento e atuação do jato;
- integração experimental com Arduino e servomotores.

Versão oficial atual: **v0.3.0**.

## Estrutura principal

```text
BLAZE/
├── vision/
│   ├── cylinders_detect/
│   ├── fire_detect/
│   ├── datasets/
│   ├── parameters_setups/
│   └── results/
├── jet_automation/
├── scripts/
├── src/
├── requirements.txt
└── README.md
```

## Instalação básica

Clone o repositório:

```bash
git clone https://github.com/Ekamitani/BLAZE.git
cd BLAZE
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Linux:

```bash
source .venv/bin/activate
```

Ative o ambiente no Windows:

```powershell
.venv\Scripts\activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para instruções detalhadas, consulte o [Guia de uso local](GUIA_USO_LOCAL.md).

## Uso no Visual Studio Code

1. Abra a pasta `BLAZE` no Visual Studio Code.
2. Instale as extensões Python e Jupyter.
3. Selecione o interpretador Python da pasta `.venv`.
4. Abra o notebook desejado.
5. Selecione o mesmo ambiente como kernel do Jupyter.

## Detecção clássica de cilindros

Notebook principal de desenvolvimento e treinamento:

[`detector_cilindros_hough_hog_svm_interativo_v2_pares_retas_corrigido.ipynb`](vision/cylinders_detect/classic_vision/detector_cilindros_hough_hog_svm_interativo_v2_pares_retas_corrigido.ipynb)

Notebook comparativo por janela deslizante:

[`detector_cilindros_hog_svm_janela_deslizante_teste_comparativo.ipynb`](vision/cylinders_detect/classic_vision/detector_cilindros_hog_svm_janela_deslizante_teste_comparativo.ipynb)

Runtime interativo para webcam:

[`runtime_classico_v2_dropdown_setup_sliders_webcam.ipynb`](vision/cylinders_detect/classic_vision/runtime_classico_v2_dropdown_setup_sliders_webcam.ipynb)

Núcleo Python do runtime:

[`_runtime_classico_v2_core.py`](vision/cylinders_detect/classic_vision/_runtime_classico_v2_core.py)

Execução rápida no Linux:

```bash
./rodar_runtime_classico_v2.sh
```

O runtime utiliza por padrão os parâmetros registrados nos metadados do modelo treinado.

## Detecção de cilindros com YOLO

Notebook principal:

[`detector_cilindros_yolo_interativo_setups_tempo_memoria.ipynb`](vision/cylinders_detect/YOLO/detector_cilindros_yolo_interativo_setups_tempo_memoria.ipynb)

Runtime interativo para webcam:

[`runtime_yolo_dropdown_setup_sliders_webcam.ipynb`](vision/cylinders_detect/YOLO/runtime_yolo_dropdown_setup_sliders_webcam.ipynb)

Modelo oficial:

[`yolo_cylinders_best.pt`](vision/results/yolo_cylinder_detector/weights/yolo_cylinders_best.pt)

## Monitoramento monocular de fogo em RGB

Notebook principal:

[`fire_monitor_RGB_monocular_yolo_arduino.ipynb`](vision/fire_detect/fire_monitor_RGB_monocular/fire_monitor_RGB_monocular_yolo_arduino.ipynb)

O módulo realiza detecção de regiões de fogo, planejamento de trajetória do jato e integração experimental com Arduino.

## Visão térmica estéreo e automação do jato

Notebook principal:

[`thermal_stereo_vision_jet_automation.ipynb`](vision/fire_detect/jet_automation/thermal_stereo_vision_jet_automation.ipynb)

Arquivo de calibração estéreo:

[`stereo_calibration2.npz`](vision/fire_detect/jet_automation/parameters_setups/stereo_calibration2.npz)

O módulo combina detecção térmica, estimativa de profundidade e planejamento da atuação do jato de água.

## Simulador de combate a incêndio

Notebook principal:

[`firefighting_simulator.ipynb`](jet_automation/simulation/firefighting_simulator/firefighting_simulator.ipynb)

Documentação específica:

[`README.md`](jet_automation/simulation/firefighting_simulator/README.md)

O simulador permite testar estratégias de detecção, priorização de focos e planejamento da trajetória do jato.

## Guias do projeto

- [Fluxo de trabalho com Git](GUIA_FLUXO_GIT.md)
- [Navegação pela estrutura do projeto](GUIA_NAVEGACAO_PROJETO.md)
- [Setups paramétricos](GUIA_SETUPS_PARAMETRICOS.md)
- [Uso local](GUIA_USO_LOCAL.md)
- [Instalação dos dados](INSTALACAO_DADOS.md)
- [Contribuição](CONTRIBUTING.md)

## Setups paramétricos

Os parâmetros dos módulos podem ser organizados em setups reutilizáveis.

Os setups oficiais ficam em pastas `parameters_setups/official`, enquanto configurações pessoais podem ser mantidas separadamente.

Consulte o [Guia de setups paramétricos](GUIA_SETUPS_PARAMETRICOS.md) para criar, salvar, comparar e promover configurações.

## Modelos e dados

Os modelos oficiais necessários para execução estão versionados no repositório.

Datasets completos, vídeos e arquivos grandes de resultados devem ser instalados separadamente.

Consulte o [Guia de instalação dos dados](INSTALACAO_DADOS.md) para conferir a estrutura esperada.

## Contribuição e versionamento

As alterações devem ser desenvolvidas em uma branch (linha de desenvolvimento) própria e integradas por meio de PR (solicitação de integração).

Antes de contribuir, consulte:

- [Guia de contribuição](CONTRIBUTING.md)
- [Guia de fluxo com Git](GUIA_FLUXO_GIT.md)

As versões oficiais do projeto são identificadas por tags (marcadores de versão) e publicadas na área de Releases do GitHub.

## Licença

Consulte o arquivo de licença do repositório para verificar as condições de uso, modificação e distribuição.

## Projeto BLAZE

Desenvolvido no contexto de atividades acadêmicas e voluntárias da Universidade Federal de Santa Catarina, campus Joinville, com foco em aplicações de robótica e visão computacional para apoio ao combate a incêndios.
