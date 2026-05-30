from pathlib import Path
import argparse
import copy
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "cilindros_parametricos": {
        "label": "Detector de cilindros - setups paramétricos gerais",
        "official": PROJECT_ROOT / "vision/parameters_setups/official/cylinders_detect/setups_parametricos.json",
        "user": PROJECT_ROOT / "vision/parameters_setups/user/cylinders_detect/setups_parametricos_user.json",
    },
    "cilindros_refinamento": {
        "label": "Detector de cilindros - setups de refinamento/expansão",
        "official": PROJECT_ROOT / "vision/parameters_setups/official/cylinders_detect/setups_refinamento_expansao.json",
        "user": PROJECT_ROOT / "vision/parameters_setups/user/cylinders_detect/setups_refinamento_expansao_user.json",
    },
    "simulador_incendio": {
        "label": "Simulador de combate - setups de simulação",
        "official": PROJECT_ROOT / "jet_automation/parameters_setups/official/firefighting_simulator/setups_simulacao_incendio.json",
        "user": PROJECT_ROOT / "jet_automation/parameters_setups/user/firefighting_simulator/setups_simulacao_incendio_user.json",
    },
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

def get_setups(data):
    setups = data.get("setups", {})
    if isinstance(setups, dict):
        return setups
    raise TypeError("Este script espera que a chave 'setups' seja um dicionário.")


def update_timestamp(data):
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "atualizado_em" in data:
        data["atualizado_em"] = agora
    elif "updated_at" in data:
        data["updated_at"] = agora


def promover_setup(tipo, setup_id, overwrite=False):
    if tipo not in CONFIGS:
        tipos = ", ".join(CONFIGS.keys())
        raise ValueError(f"Tipo inválido: {tipo}. Tipos disponíveis: {tipos}")

    cfg = CONFIGS[tipo]
    official_path = cfg["official"]
    user_path = cfg["user"]

    print("=" * 100)
    print(cfg["label"])
    print("=" * 100)
    print(f"Tipo: {tipo}")
    print(f"Setup ID: {setup_id}")
    print(f"Origem local : {user_path.relative_to(PROJECT_ROOT)}")
    print(f"Destino oficial: {official_path.relative_to(PROJECT_ROOT)}")

    user_data = load_json(user_path)
    official_data = load_json(official_path)

    user_setups = get_setups(user_data)
    official_setups = get_setups(official_data)

    if setup_id not in user_setups:
        disponiveis = ", ".join(sorted(user_setups.keys())) or "nenhum"
        raise KeyError(f"Setup '{setup_id}' não encontrado no arquivo local. Disponíveis: {disponiveis}")

    if setup_id in official_setups and not overwrite:
        raise RuntimeError(
            f"O setup '{setup_id}' já existe no arquivo oficial. "
            "Use --overwrite para sobrescrever."
        )

    official_setups[setup_id] = copy.deepcopy(user_setups[setup_id])
    official_data["setups"] = official_setups
    update_timestamp(official_data)
    save_json(official_path, official_data)

    print()
    print("Setup promovido com sucesso para official/.")
    print()
    print("Próximos passos recomendados:")
    print("  1. Testar o notebook usando o setup oficial promovido.")
    print("  2. Conferir o Git:")
    print("     git status --short")
    print("  3. Fazer commit e push:")
    print(f"     git add {official_path.relative_to(PROJECT_ROOT)}")
    print(f"     git commit -m \"Promove setup oficial {setup_id}\"")
    print("     git push")

def main():
    parser = argparse.ArgumentParser(
        description="Promove um setup local user/ para setup oficial official/."
    )
    parser.add_argument(
        "--tipo",
        required=True,
        choices=sorted(CONFIGS.keys()),
        help="Tipo de setup a promover."
    )
    parser.add_argument(
        "--id",
        required=True,
        dest="setup_id",
        help="ID do setup a promover. Exemplo: S005, R002."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescrever um setup oficial existente com o mesmo ID."
    )

    args = parser.parse_args()
    promover_setup(args.tipo, args.setup_id, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"ERRO: {exc}")
        raise SystemExit(1)
