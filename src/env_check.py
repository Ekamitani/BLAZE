from pathlib import Path
import importlib.util
import subprocess
import sys
import platform


REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "PIL": "pillow",
    "ipywidgets": "ipywidgets",
    "joblib": "joblib",
    "tqdm": "tqdm",
}


def is_installed(import_name):
    """
    Verifica se um módulo Python pode ser importado.
    """
    return importlib.util.find_spec(import_name) is not None


def check_environment(project_root=None, auto_install=False):
    """
    Verifica bibliotecas principais do projeto.

    Se auto_install=True, tenta instalar as dependências usando requirements.txt.
    """
    project_root = Path(project_root or Path.cwd()).resolve()
    requirements_file = project_root / "requirements.txt"

    print("Sistema operacional:", platform.system())
    print("Python:", sys.version)
    print("Executável Python:", sys.executable)

    missing = []

    for import_name, pip_name in REQUIRED_IMPORTS.items():
        if not is_installed(import_name):
            missing.append(pip_name)

    if not missing:
        print("Todas as bibliotecas principais foram encontradas.")
        return True

    print("Bibliotecas ausentes:")
    for pkg in missing:
        print(f"- {pkg}")

    if auto_install:
        if not requirements_file.exists():
            print("requirements.txt não encontrado.")
            return False

        print("Instalando dependências a partir de requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-U", "pip"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("Instalação concluída. Recomenda-se reiniciar o kernel.")
        return True

    print("Instalação automática desativada.")
    print("Para instalar manualmente, execute:")
    print("python -m pip install -r requirements.txt")
    return False
