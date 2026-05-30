from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    (
        'cilindros_parametricos',
        'Detector de cilindros - setups paramétricos gerais',
        PROJECT_ROOT / 'vision/parameters_setups/official/cylinders_detect/setups_parametricos.json',
        PROJECT_ROOT / 'vision/parameters_setups/user/cylinders_detect/setups_parametricos_user.json',
    ),
    (
        'cilindros_refinamento',
        'Detector de cilindros - setups de refinamento/expansão',
        PROJECT_ROOT / 'vision/parameters_setups/official/cylinders_detect/setups_refinamento_expansao.json',
        PROJECT_ROOT / 'vision/parameters_setups/user/cylinders_detect/setups_refinamento_expansao_user.json',
    ),
    (
        'simulador_incendio',
        'Simulador de combate - setups de simulação',
        PROJECT_ROOT / 'jet_automation/parameters_setups/official/firefighting_simulator/setups_simulacao_incendio.json',
        PROJECT_ROOT / 'jet_automation/parameters_setups/user/firefighting_simulator/setups_simulacao_incendio_user.json',
    ),
]


def load_json(path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_setups(data):
    if not isinstance(data, dict):
        return {}

    setups = data.get('setups', {})

    if isinstance(setups, dict):
        return setups

    if isinstance(setups, list):
        result = {}
        for i, item in enumerate(setups):
            if isinstance(item, dict):
                name = item.get('name') or item.get('id') or item.get('setup_id') or item.get('nome') or f'item_{i}'
                result[str(name)] = item
        return result

    return {}


def normalize(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def compare_pair(key, title, official_path, user_path, verbose=False):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)
    print(f'Identificador: {key}')
    print(f'Oficial: {official_path.relative_to(PROJECT_ROOT)}')
    print(f'Local  : {user_path.relative_to(PROJECT_ROOT)}')

    official_data = load_json(official_path)
    user_data = load_json(user_path)

    if official_data is None:
        print('ATENÇÃO: arquivo oficial não encontrado.')
        official = {}
    else:
        official = extract_setups(official_data)

    if user_data is None:
        print('ATENÇÃO: arquivo local não encontrado.')
        user = {}
    else:
        user = extract_setups(user_data)

    official_keys = set(official.keys())
    user_keys = set(user.keys())

    only_user = sorted(user_keys - official_keys)
    only_official = sorted(official_keys - user_keys)
    changed = sorted(k for k in (official_keys & user_keys) if normalize(official[k]) != normalize(user[k]))
    equal = sorted(k for k in (official_keys & user_keys) if normalize(official[k]) == normalize(user[k]))

    print(f'Quantidade oficial: {len(official_keys)}')
    print(f'Quantidade local  : {len(user_keys)}')

    if verbose:
        print('\nSetups oficiais:')
        for name in sorted(official_keys):
            print(f'  [official] {name}')
        print('\nSetups locais:')
        for name in sorted(user_keys):
            print(f'  [user] {name}')

    print('\nResumo da comparação:')
    print(f'  Iguais                    : {len(equal)}')
    print(f'  Só no local user/         : {len(only_user)}')
    print(f'  Só no oficial official/   : {len(only_official)}')
    print(f'  Mesmo ID, parâmetros dif. : {len(changed)}')

    if only_user:
        print('\nSetups locais candidatos a promover para official/:')
        for name in only_user:
            print(f'  + {name}')

    if changed:
        print('\nSetups com mesmo ID, mas parâmetros diferentes:')
        for name in changed:
            print(f'  ! {name}')

    if only_official:
        print('\nSetups que existem no official/, mas não existem no user/:')
        for name in only_official:
            print(f'  - {name}')

    if not only_user and not changed:
        print('\nNenhum setup local novo ou alterado para exportar neste arquivo.')


def main():
    parser = argparse.ArgumentParser(description='Compara setups locais user/ com setups oficiais official/.')
    parser.add_argument('--verbose', action='store_true', help='Lista todos os setups de cada arquivo.')
    args = parser.parse_args()

    print('Projeto:', PROJECT_ROOT)
    for pair in PAIRS:
        compare_pair(*pair, verbose=args.verbose)


if __name__ == '__main__':
    main()
