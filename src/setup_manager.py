from pathlib import Path
import json
import subprocess


def update_repository(project_root):
    """
    Tenta atualizar o repositório local com git pull --ff-only.

    Se a pasta não for um repositório Git ou se o Git não estiver instalado,
    a função apenas informa o problema e permite que o notebook continue.
    """
    project_root = Path(project_root)

    if not (project_root / ".git").exists():
        print("Esta pasta não é um repositório Git. Atualização automática ignorada.")
        return False

    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            print("Repositório atualizado com sucesso.")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True

        print("Não foi possível atualizar automaticamente com git pull --ff-only.")
        if result.stderr.strip():
            print(result.stderr.strip())
        return False

    except FileNotFoundError:
        print("Git não encontrado no sistema. Atualização automática ignorada.")
        return False


def list_setups(official_dir, user_dir, subfolder=None):
    """
    Lista setups oficiais e locais.

    Parâmetros
    ----------
    official_dir : str ou Path
        Pasta base de setups oficiais.
    user_dir : str ou Path
        Pasta base de setups locais do usuário.
    subfolder : str, opcional
        Subpasta específica do sistema, por exemplo:
        - cylinders_detect
        - firefighting_simulator
    """
    official_dir = Path(official_dir)
    user_dir = Path(user_dir)

    if subfolder:
        official_dir = official_dir / subfolder
        user_dir = user_dir / subfolder

    user_dir.mkdir(parents=True, exist_ok=True)

    setups = []

    if official_dir.exists():
        for file in sorted(official_dir.glob("*.json")):
            setups.append({
                "name": file.stem,
                "type": "official",
                "path": file
            })

    if user_dir.exists():
        for file in sorted(user_dir.glob("*.json")):
            setups.append({
                "name": file.stem,
                "type": "user",
                "path": file
            })

    return setups


def load_setup(path):
    """
    Carrega um setup JSON.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_user_setup(user_dir, setup_name, parameters, subfolder=None, overwrite=False):
    """
    Salva um setup local do usuário.

    Importante:
    esta função nunca salva em pastas oficiais.
    """
    user_dir = Path(user_dir)

    if subfolder:
        user_dir = user_dir / subfolder

    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = str(setup_name).strip()
    safe_name = safe_name.replace("/", "_").replace("\\", "_")

    if not safe_name:
        raise ValueError("O nome do setup não pode ficar vazio.")

    output_file = user_dir / f"{safe_name}.json"

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"O setup local '{output_file.name}' já existe. "
            "Use overwrite=True para sobrescrever."
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=4, ensure_ascii=False)

    print(f"Setup local salvo em: {output_file}")
    return output_file


def print_setups(setups):
    """
    Mostra a lista de setups encontrados.
    """
    if not setups:
        print("Nenhum setup encontrado.")
        return

    print("Setups encontrados:")
    for item in setups:
        print(f"- [{item['type']}] {item['name']} -> {item['path']}")
