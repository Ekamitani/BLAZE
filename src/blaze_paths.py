from pathlib import Path
import os


def find_project_root(start=None):
    """
    Encontra a raiz do projeto BLAZE.

    A busca sobe pelas pastas a partir do diretório atual até encontrar:
    - uma pasta .git
    - ou o arquivo requirements.txt
    - ou uma pasta vision e uma pasta jet_automation

    Isso evita dependência de caminhos fixos do servidor Jupyter da UFSC.
    """
    start = Path(start or os.getcwd()).resolve()

    for folder in [start, *start.parents]:
        has_git = (folder / ".git").exists()
        has_requirements = (folder / "requirements.txt").exists()
        has_blaze_structure = (folder / "vision").exists() and (folder / "jet_automation").exists()

        if has_git or has_requirements or has_blaze_structure:
            return folder

    return start


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"

VISION_DIR = PROJECT_ROOT / "vision"
VISION_DATASETS_DIR = VISION_DIR / "datasets"
VISION_RESULTS_DIR = VISION_DIR / "results"
VISION_PARAMETERS_DIR = VISION_DIR / "parameters_setups"
VISION_OFFICIAL_SETUPS_DIR = VISION_PARAMETERS_DIR / "official"
VISION_USER_SETUPS_DIR = VISION_PARAMETERS_DIR / "user"

JET_AUTOMATION_DIR = PROJECT_ROOT / "jet_automation"
JET_RESULTS_DIR = JET_AUTOMATION_DIR / "results"
JET_PARAMETERS_DIR = JET_AUTOMATION_DIR / "parameters_setups"
JET_OFFICIAL_SETUPS_DIR = JET_PARAMETERS_DIR / "official"
JET_USER_SETUPS_DIR = JET_PARAMETERS_DIR / "user"


def ensure_base_dirs():
    """
    Garante que as principais pastas locais existam.
    """
    dirs = [
        VISION_RESULTS_DIR,
        VISION_USER_SETUPS_DIR,
        JET_RESULTS_DIR,
        JET_USER_SETUPS_DIR,
    ]

    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)


def print_project_paths():
    """
    Mostra os principais caminhos detectados.
    """
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"VISION_DATASETS_DIR: {VISION_DATASETS_DIR}")
    print(f"VISION_RESULTS_DIR: {VISION_RESULTS_DIR}")
    print(f"VISION_OFFICIAL_SETUPS_DIR: {VISION_OFFICIAL_SETUPS_DIR}")
    print(f"VISION_USER_SETUPS_DIR: {VISION_USER_SETUPS_DIR}")
    print(f"JET_RESULTS_DIR: {JET_RESULTS_DIR}")
    print(f"JET_OFFICIAL_SETUPS_DIR: {JET_OFFICIAL_SETUPS_DIR}")
    print(f"JET_USER_SETUPS_DIR: {JET_USER_SETUPS_DIR}")
