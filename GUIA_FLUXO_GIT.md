# Guia de fluxo Git do projeto BLAZE

Este guia explica como usar Git e GitHub no projeto BLAZE, especialmente a diferença entre main, dev, branches de trabalho e Pull Requests.

## 1. Ideia geral

O projeto usa um fluxo simples:

    main -> versão estável
    dev  -> desenvolvimento contínuo
    branches específicas -> mudanças maiores ou experimentais

A branch main deve representar uma versão funcional e confiável do projeto.

A branch dev deve ser usada para desenvolvimento, testes e melhorias antes de incorporar mudanças à main.

Branches específicas podem ser criadas a partir da dev para trabalhos maiores, como desenvolvimento do detector YOLO, modularização ou reestruturação de código.

## 2. Branch main

A branch main é a linha estável do projeto.

Ela deve ser usada para:

    - versões validadas;
    - releases e tags;
    - documentação estável;
    - setups oficiais já testados;
    - correções simples e seguras.

Evite desenvolver mudanças grandes diretamente na main.

## 3. Branch dev

A branch dev é a linha de desenvolvimento do projeto.

Use a dev para:

    - desenvolvimento do detector YOLO;
    - mudanças grandes em notebooks;
    - modularização de código em src/;
    - reorganização de estrutura;
    - testes que ainda não devem ir para a main.

Para trabalhar na dev:

    git switch dev
    git pull --ff-only

Depois de editar e testar:

    git status --short
    git add caminho/do/arquivo
    git commit -m "Mensagem objetiva da alteração"
    git push

## 4. Branches específicas

Para mudanças maiores, é recomendado criar uma branch específica a partir da dev.

Exemplos de nomes:

    yolo-cilindros
    yolo-fogo
    modularizacao-detector-cilindros
    ajuste-sistema-setups

Exemplo de criação:

    git switch dev
    git pull --ff-only
    git switch -c yolo-cilindros
    git push -u origin yolo-cilindros

Nessa branch específica, você pode desenvolver e testar sem afetar diretamente a main.

## 5. Pull Requests

Pull Request, ou PR, é uma solicitação para incorporar uma branch em outra.

Exemplos:

    yolo-cilindros -> dev
    dev -> main

Para contribuições externas, o ideal é que a pessoa trabalhe em uma branch ou fork e abra um PR para revisão.

Você, como responsável pelo repositório, revisa o PR e decide se aceita, pede ajustes ou rejeita.

## 6. Setups paramétricos e Git

Setups locais salvos pelos notebooks ficam nas pastas user/ e não são enviados ao GitHub.

Exemplos:

    vision/parameters_setups/user/
    jet_automation/parameters_setups/user/

Setups oficiais ficam nas pastas official/ e são versionados pelo GitHub.

Exemplos:

    vision/parameters_setups/official/
    jet_automation/parameters_setups/official/

Para comparar setups locais e oficiais:

    python scripts/comparar_setups.py --verbose

Para promover um setup local para oficial:

    python scripts/promover_setup.py --tipo cilindros_parametricos --id S005

Depois de promover e testar, envie o setup oficial para o GitHub:

    git status --short
    git add caminho/do/setup/oficial.json
    git commit -m "Promove setup oficial S005"
    git push

## 7. Tags e releases

Tags marcam versões importantes do projeto.

Exemplos já usados:

    v0.1.0 -> notebooks validados
    v0.1.1 -> guia de instalação dos dados
    v0.1.2 -> documentação complementar completa
    v0.1.3 -> automação de setups e documentação dos módulos

Para criar uma nova tag:

    git tag -a v0.1.4 -m "Mensagem da versão"
    git push origin v0.1.4

Depois, a release pode ser criada no GitHub a partir da tag.

## 8. Fluxo diário recomendado

Antes de começar qualquer alteração:

    git status
    git pull --ff-only

Para trabalhar em desenvolvimento:

    git switch dev
    git pull --ff-only

Depois de editar e testar:

    git status --short
    git diff --stat
    git add caminho/do/arquivo
    git commit -m "Mensagem objetiva"
    git push

Para voltar para a versão estável:

    git switch main
    git pull --ff-only

## 9. Regra prática

Use main para estabilidade.
Use dev para desenvolvimento.
Use branches específicas para mudanças grandes.
Use Pull Requests para revisar antes de incorporar mudanças importantes.
