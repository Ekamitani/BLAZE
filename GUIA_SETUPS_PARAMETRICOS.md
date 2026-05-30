# Guia de setups paramétricos do projeto BLAZE

Este guia explica como salvar, organizar, exportar e compartilhar setups paramétricos dos notebooks do projeto BLAZE.

## 1. Tipos de setup

O projeto trabalha com dois tipos principais de setup: setups oficiais e setups locais do usuário.

### Setups oficiais

São setups versionados no GitHub e compartilhados com todos que usam o repositório.

Eles ficam em:

    vision/parameters_setups/official/
    jet_automation/parameters_setups/official/

Exemplos:

    vision/parameters_setups/official/cylinders_detect/setups_parametricos.json
    vision/parameters_setups/official/cylinders_detect/setups_refinamento_expansao.json
    jet_automation/parameters_setups/official/firefighting_simulator/setups_simulacao_incendio.json

Esses arquivos devem ser usados para configurações consideradas estáveis, úteis ou importantes para outras pessoas.

### Setups locais do usuário

São setups criados durante o uso do notebook em uma máquina específica.

Eles ficam em:

    vision/parameters_setups/user/
    jet_automation/parameters_setups/user/

Essas pastas são ignoradas pelo GitHub pelo arquivo .gitignore.

Isso evita que cada teste local, ajuste temporário ou experimento pessoal seja enviado automaticamente para o repositório.

## 2. Regra prática

Quando você salva um setup pelo notebook, ele normalmente vai para a pasta user/.

Ou seja:

    setup salvo no notebook -> setup local -> não vai para o GitHub

Para compartilhar um setup com outras pessoas, você precisa promovê-lo para setup oficial.

## 3. Como transformar um setup local em setup oficial

O processo recomendado é:

    1. Salvar o setup normalmente pelo notebook.
    2. Conferir o arquivo salvo na pasta user/.
    3. Copiar o conteúdo relevante para a pasta official/.
    4. Testar o notebook usando esse setup.
    5. Fazer commit e push para o GitHub.

Depois disso, outras pessoas poderão obter o setup com:

    git pull --ff-only

## 4. Detector clássico de cilindros

Os setups locais do detector de cilindros ficam em:

    vision/parameters_setups/user/cylinders_detect/

Os setups oficiais ficam em:

    vision/parameters_setups/official/cylinders_detect/

Arquivos principais:

    setups_parametricos.json
    setups_refinamento_expansao.json

O arquivo setups_parametricos.json guarda setups gerais do pipeline, como pré-processamento, Hough, Canny, filtros e parâmetros principais.

O arquivo setups_refinamento_expansao.json guarda setups específicos do refinamento, expansão, validação geométrica e estratégias orientadas por linhas/arcos.

## 5. Simulador de combate a incêndio

Os setups locais do simulador ficam em:

    jet_automation/parameters_setups/user/firefighting_simulator/

Os setups oficiais ficam em:

    jet_automation/parameters_setups/official/firefighting_simulator/

Arquivo principal:

    setups_simulacao_incendio.json

## 6. Como enviar um setup oficial para o GitHub

Depois de atualizar um arquivo em official/, rode:

    git status

Se aparecer, por exemplo:

    M vision/parameters_setups/official/cylinders_detect/setups_parametricos.json

faça:

    git add vision/parameters_setups/official/cylinders_detect/setups_parametricos.json
    git commit -m "Adiciona setup oficial do detector de cilindros"
    git push

Para setups do simulador:

    git add jet_automation/parameters_setups/official/firefighting_simulator/setups_simulacao_incendio.json
    git commit -m "Adiciona setup oficial do simulador"
    git push

## 7. Como outra pessoa recebe os setups novos

No computador da outra pessoa, dentro da pasta BLAZE:

    git pull --ff-only

Depois disso, os setups oficiais novos estarão disponíveis no notebook.

## 8. O que não deve ser enviado como setup oficial

Evite enviar para official/:

    - testes incompletos;
    - setups temporários;
    - setups com nomes confusos;
    - configurações que só funcionam em uma imagem específica;
    - ajustes feitos apenas para depuração.

Use a pasta user/ para experimentos locais.

## 9. Boas práticas para nomear setups

Use nomes claros, por exemplo:

    cilindros_hough_conservador_v1
    cilindros_refino_linhas_fortes_v2
    simulador_trajetoria_offset_prioridade_foco_v1
    simulador_cenario_chama_alta_v1

Evite nomes genéricos como:

    teste
    novo
    setup1
    final
    final2

## 10. Fluxo recomendado para você

Antes de começar:

    git pull --ff-only

Depois de testar e decidir que um setup deve ser compartilhado:

    git status
    git add caminho/do/setup/oficial.json
    git commit -m "Adiciona setup oficial ..."
    git push

Depois, peça para a outra pessoa atualizar:

    git pull --ff-only
