#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime V2 em VS Code para detecção clássica de cilindros com webcam.

Este arquivo executa somente a inferência em tempo real:
- não treina modelo;
- não abre painéis Jupyter;
- carrega setups JSON salvos no projeto;
- carrega o modelo HOG/SVM treinado localmente;
- mostra apenas a BBox final da detecção na tela;
- permite ajustar resolução, taxa de atualização e score mínimo por sliders OpenCV.

Uso básico:
    python vision/cylinders_detect/classic_vision/_runtime_classico_v2_core.py

Listar setups:
    python vision/cylinders_detect/classic_vision/_runtime_classico_v2_core.py --list-setups

Escolher setup direto:
    python vision/cylinders_detect/classic_vision/_runtime_classico_v2_core.py --setup S001

Sair:
    pressione q na janela da webcam.
"""

from pathlib import Path
from collections import OrderedDict
import argparse
import os
import sys
import shutil
import cv2
import json
import time
import math
import hashlib
import traceback
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception as e:
    raise ImportError("Instale o joblib: pip install joblib") from e


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ============================================================
# Caminhos portáteis do projeto BLAZE
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

for _candidate in [CURRENT_DIR, *CURRENT_DIR.parents]:
    if (_candidate / 'src' / 'blaze_paths.py').exists():
        PROJECT_ROOT = _candidate
        break
else:
    raise RuntimeError(
        "Não foi possível localizar a raiz do projeto BLAZE. "
        "Execute este script a partir de dentro do repositório BLAZE."
    )

SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blaze_paths import (
    VISION_DATASETS_DIR,
    VISION_RESULTS_DIR,
    VISION_OFFICIAL_SETUPS_DIR,
    VISION_USER_SETUPS_DIR,
    ensure_base_dirs,
)

ensure_base_dirs()

RESULTS_DIR = VISION_RESULTS_DIR
CLASSICAL_DIR = RESULTS_DIR / 'classical_cylinder_detector'
CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CLASSICAL_DIR / 'modelo_hog_svm_cilindros_v2.joblib'
METADATA_PATH = CLASSICAL_DIR / 'metadata_modelo_v2.json'
HISTORICO_PATH = CLASSICAL_DIR / 'historico_treinos_v2.csv'
HISTORY_PATH = HISTORICO_PATH

OFFICIAL_PARAM_SETUPS_PATH = VISION_OFFICIAL_SETUPS_DIR / 'cylinders_detect' / 'setups_parametricos_v2.json'
USER_PARAM_SETUPS_PATH = VISION_USER_SETUPS_DIR / 'cylinders_detect' / 'setups_parametricos_v2_user.json'

OFFICIAL_REFINE_SETUPS_PATH = VISION_OFFICIAL_SETUPS_DIR / 'cylinders_detect' / 'setups_refinamento_expansao_v2_nao_usado.json'
USER_REFINE_SETUPS_PATH = VISION_USER_SETUPS_DIR / 'cylinders_detect' / 'setups_refinamento_expansao_v2_nao_usado_user.json'

DATASET_DIR = VISION_DATASETS_DIR / 'cylinders' / 'CylinDeRS-1'
TRAIN_IMG_DIR = DATASET_DIR / 'train' / 'images'
TRAIN_LABEL_DIR = DATASET_DIR / 'train' / 'labels'
TEST_IMG_DIR = DATASET_DIR / 'test' / 'images'
TEST_LABEL_DIR = DATASET_DIR / 'test' / 'labels'

# Arquivos usados apenas por algumas funções herdadas do notebook.
SELECAO_TREINO_PATH = CLASSICAL_DIR / 'imagens_selecionadas_treino.json'
SELECAO_TESTE_PATH = CLASSICAL_DIR / 'imagens_selecionadas_teste.json'
ZOOMOUT_TREINO_PATH = CLASSICAL_DIR / 'imagens_zoomout_treino.json'
ZOOMOUT_TESTE_PATH = CLASSICAL_DIR / 'imagens_zoomout_teste.json'
ZOOMOUT_CONFIG_PATH = CLASSICAL_DIR / 'zoomout_config.json'
ZOOMOUT_DATASET_DIR = CLASSICAL_DIR / 'dataset_zoomout_aplicado'
ZOOMOUT_MANIFEST_PATH = CLASSICAL_DIR / 'dataset_zoomout_manifest.json'
ZOOMOUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)


# Fallbacks para funções herdadas que foram escritas originalmente para Jupyter.
def display(*args, **kwargs):
    return None

def HTML(x):
    return x

def clear_output(*args, **kwargs):
    return None



# ============================================================
# 2. Funções utilitárias gerais e leitura do dataset YOLO
# ============================================================

def caminhos_dataset(dataset_dir):
    """Retorna os caminhos padronizados para um dataset no formato YOLO train/test."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    return {
        'dataset_dir': dataset_dir,
        'train_img_dir': dataset_dir / 'train' / 'images',
        'train_label_dir': dataset_dir / 'train' / 'labels',
        'test_img_dir': dataset_dir / 'test' / 'images',
        'test_label_dir': dataset_dir / 'test' / 'labels',
    }


def label_path_para_imagem(img_path, img_dir, label_dir):
    """Converte caminho de imagem em caminho esperado do label YOLO correspondente."""
    img_path = Path(img_path)
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    try:
        rel = img_path.relative_to(img_dir)
    except ValueError:
        rel = Path(img_path.name)
    return (label_dir / rel).with_suffix('.txt')


def contar_labels_yolo(label_path):
    """Conta apenas linhas YOLO válidas e finitas em um arquivo de label.

    Alguns arquivos podem conter valores ausentes/NaN. Esses registros são ignorados
    para evitar erros posteriores na conversão para bounding boxes.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return 0

    n = 0
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = float(parts[0])
                vals = list(map(float, parts[1:5]))
            except Exception:
                continue
            if not (np.isfinite(cls) and all(np.isfinite(v) for v in vals)):
                continue
            xc, yc, bw, bh = vals
            if bw <= 0 or bh <= 0:
                continue
            n += 1
    return n


def listar_imagens_split(split, img_dir, label_dir):
    """Lista imagens de um split e associa cada uma ao arquivo de label YOLO."""
    img_dir = Path(img_dir).expanduser().resolve()
    label_dir = Path(label_dir).expanduser().resolve()

    rows = []
    if not img_dir.exists():
        return rows

    for p in sorted(img_dir.rglob('*')):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue

        lab = label_path_para_imagem(p, img_dir, label_dir)
        n_obj = contar_labels_yolo(lab)
        rows.append({
            'split': split,
            'path': str(p),
            'arquivo': p.name,
            'stem': p.stem,
            'label_path': str(lab),
            'label_existe': lab.exists(),
            'n_obj': int(n_obj),
            'classe': 'com_anotacao' if n_obj > 0 else 'sem_anotacao'
        })

    return rows


def listar_imagens_dataset(dataset_dir):
    """Lista imagens de train e test a partir da raiz do CylinDeRS-1."""
    paths = caminhos_dataset(dataset_dir)

    rows = []
    rows.extend(listar_imagens_split('train', paths['train_img_dir'], paths['train_label_dir']))
    rows.extend(listar_imagens_split('test', paths['test_img_dir'], paths['test_label_dir']))

    return pd.DataFrame(rows, columns=[
        'split', 'path', 'arquivo', 'stem', 'label_path', 'label_existe', 'n_obj', 'classe'
    ])


def ler_imagem_bgr(path_img):
    """Lê imagem com OpenCV em BGR."""
    img = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Não foi possível carregar a imagem: {path_img}')
    return img


def bgr_para_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    return x1, y1, x2, y2


def resumo_dataset(df):
    if df is None or len(df) == 0:
        return 'Nenhuma imagem encontrada.'

    total = len(df)
    n_train = int((df['split'] == 'train').sum())
    n_test = int((df['split'] == 'test').sum())
    com_label = int(df['label_existe'].sum())
    imagens_com_obj = int((df['n_obj'] > 0).sum())
    total_obj = int(df['n_obj'].sum())

    return (
        f'Imagens: {total} | Train: {n_train} | Test: {n_test} | '
        f'Labels existentes: {com_label} | Imagens com objeto: {imagens_com_obj} | '
        f'Objetos anotados: {total_obj}'
    )


def exibir_tabela_compacta(df, max_rows=12):
    """
    Mostra uma tabela compacta com rolagem horizontal controlada,
    para evitar que ela ultrapasse a largura visual do notebook.
    """
    if df is None or len(df) == 0:
        display(HTML('<em>Nenhum registro para mostrar.</em>'))
        return

    df_show = df.tail(max_rows).copy()

    for col in df_show.columns:
        if df_show[col].dtype == object:
            df_show[col] = df_show[col].astype(str).str.slice(0, 46)

    html = df_show.to_html(index=False, escape=False)
    display(HTML(f'''
    <div style="
        max-width: 100%;
        overflow-x: auto;
        border: 1px solid #444;
        padding: 6px;
        border-radius: 8px;
    ">
        <style>
            table.dataframe {{
                font-size: 11px;
                border-collapse: collapse;
                width: max-content;
                max-width: 100%;
            }}
            table.dataframe th, table.dataframe td {{
                white-space: nowrap;
                padding: 4px 7px;
                text-align: center;
            }}
        </style>
        {html}
    </div>
    '''))


def salvar_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def carregar_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 3. Pipeline de pré-processamento com dois mapas de bordas
# ============================================================


def filtrar_componentes_borda(edge_map, p):
    """
    Remove pequenos componentes conectados do mapa binário de bordas.

    Objetivo:
        - preservar o Canny completo como diagnóstico visual;
        - gerar um segundo mapa mais limpo para Hough e geração de ROIs;
        - reduzir falsas bordas antes da formulação geométrica.

    Critérios de permanência:
        componente permanece se:
            área >= edge_min_area E maior dimensão >= edge_min_extent

    Retorna:
        edge_filtrado, info
    """
    if not bool(p.get('edge_filter_ativo', True)):
        return edge_map.copy(), {
            'enabled': False,
            'n_components': 0,
            'n_removed': 0,
            'n_kept': 0,
            'min_area': 0,
            'min_extent': 0
        }

    min_area = max(0, int(p.get('edge_min_area', 12)))
    min_extent = max(0, int(p.get('edge_min_extent', 10)))

    bin_map = (edge_map > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)

    out = np.zeros_like(edge_map)
    kept = 0
    removed = 0

    for lab in range(1, n_labels):
        x, y, w, h, area = stats[lab]
        extent = max(int(w), int(h))
        keep = (int(area) >= min_area) and (extent >= min_extent)
        if keep:
            out[labels == lab] = 255
            kept += 1
        else:
            removed += 1

    info = {
        'enabled': True,
        'n_components': int(max(0, n_labels - 1)),
        'n_removed': int(removed),
        'n_kept': int(kept),
        'min_area': int(min_area),
        'min_extent': int(min_extent)
    }
    return out, info


def aplicar_preprocessamento(img_bgr, p):
    """
    Aplica a preparação da imagem sem etapa morfológica.

    Mapas produzidos:
        - Mapa A: Canny completo, usado para diagnóstico visual;
        - Mapa B: Canny filtrado, usado para Hough e geração de ROIs.
    """
    rgb = bgr_para_rgb(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if p.get("clahe_ativo", True):
        clahe = cv2.createCLAHE(
            clipLimit=float(p.get("clahe_clip", 2.0)),
            tileGridSize=(int(p.get("clahe_grid", 8)), int(p.get("clahe_grid", 8)))
        )
        gray_eq = clahe.apply(gray)
    else:
        gray_eq = gray.copy()

    d = int(p.get("bilateral_d", 5))
    if d < 1:
        d = 1
    if d % 2 == 0:
        d += 1

    bilateral = cv2.bilateralFilter(
        gray_eq,
        d=d,
        sigmaColor=float(p.get("bilateral_sigma_color", 50)),
        sigmaSpace=float(p.get("bilateral_sigma_space", 50))
    )

    med = float(np.median(bilateral))
    sigma = float(p.get("canny_sigma", 0.33))
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))

    if upper <= lower:
        upper = min(255, lower + 20)

    aperture = int(p.get("canny_aperture", 3))
    if aperture not in [3, 5, 7]:
        aperture = 3

    canny_full = cv2.Canny(
        bilateral,
        threshold1=lower,
        threshold2=upper,
        apertureSize=aperture,
        L2gradient=bool(p.get("canny_l2gradient", True))
    )

    canny_hough, edge_filter_info = filtrar_componentes_borda(canny_full, p)

    # Gradiente usado para validar contornos curvos.
    # O valor é normalizado entre 0 e 1 para que o parâmetro de
    # gradiente mínimo seja estável entre imagens diferentes.
    grad_x = cv2.Sobel(bilateral, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(bilateral, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = grad_mag / max(1e-6, float(np.percentile(grad_mag, 99)))
    grad_norm = np.clip(grad_norm, 0.0, 1.0).astype(np.float32)

    return {
        "rgb": rgb,
        "gray": gray,
        "gray_eq": gray_eq,
        "bilateral": bilateral,
        "grad_mag": grad_norm,
        "canny": canny_full,
        "canny_full": canny_full,
        "canny_hough": canny_hough,
        "canny_filtrado": canny_hough,
        "canny_lower": lower,
        "canny_upper": upper,
        "edge_filter_info": edge_filter_info
    }


def selecionar_imagem_hog(pre, p=None):
    """
    Retorna a imagem usada para extração HOG.

    Nesta versão simplificada, o HOG/SVM usa sempre o mapa bilateral,
    porque ele preserva bordas e textura sem transformar a imagem em
    um mapa binário de bordas.
    """
    return pre["bilateral"]


# ============================================================
# 4. Hough, pareamento geométrico e geração de ROIs
# ============================================================

def detectar_linhas_hough(img_edges, p):
    """
    Detecta segmentos de reta com HoughLinesP.
    """
    lines = cv2.HoughLinesP(
        img_edges,
        rho=float(p["hough_rho"]),
        theta=np.deg2rad(float(p["hough_theta_deg"])),
        threshold=int(p["hough_threshold"]),
        minLineLength=int(p["hough_min_line_length"]),
        maxLineGap=int(p["hough_max_line_gap"])
    )

    if lines is None:
        return []

    return [tuple(map(int, l[0])) for l in lines]


def metrica_linha(line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    length = float(math.hypot(dx, dy))
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

    if length == 0:
        u = np.array([1.0, 0.0])
    else:
        u = np.array([dx / length, dy / length], dtype=float)

    return {
        "line": line,
        "length": length,
        "angle": angle,
        "center": center,
        "u": u
    }


def diff_angular_graus(a, b):
    """
    Diferença angular mínima considerando orientação de reta, não vetor.
    Retorna valor entre 0 e 90 graus.
    """
    d = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(d)


def intervalo_projetado(line, u):
    x1, y1, x2, y2 = line
    p1 = np.array([x1, y1], dtype=float)
    p2 = np.array([x2, y2], dtype=float)
    v1 = float(np.dot(p1, u))
    v2 = float(np.dot(p2, u))
    return min(v1, v2), max(v1, v2)


def razao_sobreposicao(i1, i2):
    a1, a2 = i1
    b1, b2 = i2

    inter = max(0.0, min(a2, b2) - max(a1, b1))
    len1 = max(1e-6, a2 - a1)
    len2 = max(1e-6, b2 - b1)

    return float(inter / min(len1, len2))


def encontrar_rois_por_pares_de_linhas(lines, shape_hw, p):
    """
    Procura pares de retas paralelas e gera ROIs candidatas.
    """
    h, w = shape_hw
    metricas = [metrica_linha(l) for l in lines]
    metricas = [
        m for m in metricas
        if p["line_length_min"] <= m["length"] <= p["line_length_max"]
    ]

    candidatos = []

    for i in range(len(metricas)):
        for j in range(i + 1, len(metricas)):
            m1 = metricas[i]
            m2 = metricas[j]

            adiff = diff_angular_graus(m1["angle"], m2["angle"])
            if adiff > p["pair_angle_tol_deg"]:
                continue

            # Orientação média do par.
            u = m1["u"] + m2["u"]
            if np.linalg.norm(u) < 1e-6:
                u = m1["u"]
            else:
                u = u / np.linalg.norm(u)

            # Normal à direção da reta.
            n = np.array([-u[1], u[0]], dtype=float)

            # Distância transversal entre centros.
            dist = abs(float(np.dot(m2["center"] - m1["center"], n)))
            if not (p["pair_dist_min"] <= dist <= p["pair_dist_max"]):
                continue

            # Sobreposição longitudinal.
            int1 = intervalo_projetado(m1["line"], u)
            int2 = intervalo_projetado(m2["line"], u)
            overlap = razao_sobreposicao(int1, int2)
            if overlap < p["pair_overlap_min"]:
                continue

            pts = np.array([
                [m1["line"][0], m1["line"][1]],
                [m1["line"][2], m1["line"][3]],
                [m2["line"][0], m2["line"][1]],
                [m2["line"][2], m2["line"][3]],
            ], dtype=float)

            margin = int(p["roi_margin_px"])
            x1 = int(np.floor(pts[:, 0].min())) - margin
            y1 = int(np.floor(pts[:, 1].min())) - margin
            x2 = int(np.ceil(pts[:, 0].max())) + margin
            y2 = int(np.ceil(pts[:, 1].max())) + margin

            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, w, h)

            roi_w = max(1, x2 - x1)
            roi_h = max(1, y2 - y1)
            aspect = roi_h / roi_w

            if not (p["roi_aspect_min"] <= aspect <= p["roi_aspect_max"]):
                continue

            area = roi_w * roi_h

            candidatos.append({
                "bbox": (x1, y1, x2, y2),
                "line1": m1["line"],
                "line2": m2["line"],
                "angle_diff": adiff,
                "distance": dist,
                "overlap": overlap,
                "aspect": aspect,
                "area": area
            })

    # Ordenação simples: prioriza boa sobreposição e baixa diferença angular.
    candidatos = sorted(
        candidatos,
        key=lambda c: (-c["overlap"], c["angle_diff"], abs(c["distance"] - (p["pair_dist_min"] + p["pair_dist_max"]) / 2))
    )

    return candidatos[:int(p["max_rois"])]


def desenhar_linhas(img_rgb, lines, max_lines=80):
    out = img_rgb.copy()
    for line in lines[:max_lines]:
        x1, y1, x2, y2 = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
    return out


def desenhar_rois(img_rgb, candidatos, mostrar_pares=True):
    out = img_rgb.copy()

    for idx, c in enumerate(candidatos):
        x1, y1, x2, y2 = c["bbox"]

        # ROI em azul/laranja.
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(
            out,
            f"ROI {idx+1}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 120, 0),
            1,
            cv2.LINE_AA
        )

        if mostrar_pares:
            for line in [c["line1"], c["line2"]]:
                lx1, ly1, lx2, ly2 = line
                cv2.line(out, (lx1, ly1), (lx2, ly2), (0, 220, 255), 2)

    return out

# ============================================================
# 5. HOG, labels YOLO, assinatura de parâmetros e histórico
# ============================================================

def extrair_hog_de_imagem(img_bgr, p):
    """Extrai vetor HOG de uma imagem/crop."""
    pre = aplicar_preprocessamento(img_bgr, p)
    img_hog = selecionar_imagem_hog(pre, p)

    w = int(p['hog_resize_w'])
    h = int(p['hog_resize_h'])
    img_resized = cv2.resize(img_hog, (w, h), interpolation=cv2.INTER_AREA)

    features = hog(
        img_resized,
        orientations=int(p['hog_orientations']),
        pixels_per_cell=(int(p['hog_pixels_per_cell']), int(p['hog_pixels_per_cell'])),
        cells_per_block=(int(p['hog_cells_per_block']), int(p['hog_cells_per_block'])),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)


def n_features_esperado_modelo(clf):
    """Retorna o número de atributos esperado pelo modelo treinado, quando disponível."""
    if clf is None:
        return None

    candidatos = []
    if hasattr(clf, "named_steps"):
        candidatos.extend([
            clf.named_steps.get("standardscaler"),
            clf.named_steps.get("scaler"),
            clf.named_steps.get("linearsvc"),
            clf.named_steps.get("svc"),
        ])
    if hasattr(clf, "steps"):
        candidatos.extend([step for _, step in clf.steps])
    candidatos.append(clf)

    for obj in candidatos:
        if obj is None:
            continue
        n = getattr(obj, "n_features_in_", None)
        if n is not None:
            return int(n)
    return None


def validar_features_modelo(feat, clf, contexto="ROI"):
    """Valida compatibilidade entre extrator HOG atual e modelo treinado.

    A correção limpa é retreinar o modelo quando houver incompatibilidade.
    Não preenche, corta ou mascara atributos, pois isso mistura versões de treino/runtime.
    """
    feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
    esperado = n_features_esperado_modelo(clf)
    atual = int(feat.shape[1])

    if esperado is not None and atual != esperado:
        raise RuntimeError(
            f"{contexto}: incompatibilidade de features HOG/SVM. "
            f"O runtime gerou {atual} atributos, mas o modelo espera {esperado}. "
            "Retreine o modelo no notebook atual para alinhar extrator e classificador."
        )
    return feat


def extrair_hog_visualizacao_de_imagem(img_bgr, p):
    """
    Extrai HOG e também retorna a imagem usada e a visualização dos gradientes.

    A função é usada apenas para diagnóstico visual. O treinamento continua usando
    `extrair_hog_de_imagem`, com os mesmos parâmetros.
    """
    pre = aplicar_preprocessamento(img_bgr, p)
    img_hog = selecionar_imagem_hog(pre, p)

    w = int(p['hog_resize_w'])
    h = int(p['hog_resize_h'])
    img_resized = cv2.resize(img_hog, (w, h), interpolation=cv2.INTER_AREA)

    features, hog_image = hog(
        img_resized,
        orientations=int(p['hog_orientations']),
        pixels_per_cell=(int(p['hog_pixels_per_cell']), int(p['hog_pixels_per_cell'])),
        cells_per_block=(int(p['hog_cells_per_block']), int(p['hog_cells_per_block'])),
        block_norm='L2-Hys',
        transform_sqrt=True,
        visualize=True,
        feature_vector=True
    )

    hog_image = np.asarray(hog_image, dtype=np.float32)
    if hog_image.max() > hog_image.min():
        hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())

    return img_resized, hog_image, features.astype(np.float32)


def ler_bboxes_yolo(label_path, img_w, img_h, class_filter=None):
    """
    Lê bboxes YOLO normalizadas e retorna bboxes em pixel.

    Retorna lista de dicionários:
        {'class_id': int, 'bbox': (x1, y1, x2, y2)}

    Registros inválidos, NaN, infinitos, largura/altura não positivas ou caixas
    degeneradas são ignorados. Isso evita erros do tipo
    "cannot convert float NaN to integer" durante a validação visual.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_float = float(parts[0])
                vals = list(map(float, parts[1:5]))
            except Exception:
                continue

            if not (np.isfinite(cls_float) and all(np.isfinite(v) for v in vals)):
                continue

            cls = int(cls_float)
            xc, yc, bw, bh = vals
            if bw <= 0 or bh <= 0:
                continue

            if class_filter is not None and cls not in class_filter:
                continue

            # Aceita anotações levemente fora de [0, 1], mas descarta caixas totalmente inválidas.
            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h
            if not all(np.isfinite(v) for v in [x1, y1, x2, y2]):
                continue

            # Se a caixa inteira ficou fora da imagem, ignora.
            if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
                continue

            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, img_w, img_h)

            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                boxes.append({'class_id': cls, 'bbox': (x1, y1, x2, y2)})

    return boxes


def expandir_bbox(bbox, margin_pct, img_w, img_h):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * float(margin_pct)
    my = bh * float(margin_pct)
    return clip_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, img_w, img_h)


def iou_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-9)


def bbox_aleatorio(img_w, img_h, target_w, target_h, rng):
    target_w = int(max(8, min(target_w, img_w)))
    target_h = int(max(8, min(target_h, img_h)))
    if img_w <= target_w or img_h <= target_h:
        return (0, 0, img_w, img_h)
    x1 = int(rng.integers(0, img_w - target_w))
    y1 = int(rng.integers(0, img_h - target_h))
    return (x1, y1, x1 + target_w, y1 + target_h)


def recortar_bbox(img_bgr, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return img_bgr[y1:y2, x1:x2].copy()


def extrair_strings_json(obj):
    """Extrai recursivamente strings de um JSON de seleção."""
    vals = []
    if obj is None:
        return vals
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for v in obj.values():
            vals.extend(extrair_strings_json(v))
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            vals.extend(extrair_strings_json(v))
    return vals


def carregar_chaves_selecao(path):
    """Lê um JSON de seleção e retorna chaves por caminho, nome de arquivo e stem."""
    obj = carregar_json(path, default=None)
    strings = extrair_strings_json(obj)
    chaves = set()
    for s in strings:
        p = Path(str(s))
        chaves.add(str(s))
        chaves.add(p.name)
        chaves.add(p.stem)
    return chaves



def _json_zoomout_split(split):
    """Caminho do JSON que guarda quais imagens receberão zoom out em cada split."""
    return ZOOMOUT_TREINO_PATH if split == 'train' else ZOOMOUT_TESTE_PATH


def carregar_lista_zoomout(split):
    """Lista persistente das imagens marcadas para gerar cópia com zoom out aplicado."""
    obj = carregar_json(_json_zoomout_split(split), default=[])
    strings = extrair_strings_json(obj)
    # Remove duplicatas preservando a ordem.
    vistos = set()
    saida = []
    for s in strings:
        ss = str(s)
        if ss not in vistos:
            saida.append(ss)
            vistos.add(ss)
    return saida


def carregar_chaves_zoomout(split):
    """Lê a seleção persistente de imagens que receberão zoom out e retorna chaves de comparação."""
    strings = carregar_lista_zoomout(split)
    chaves = set()
    for s in strings:
        p = Path(str(s))
        chaves.add(str(s))
        chaves.add(str(p))
        chaves.add(p.name)
        chaves.add(p.stem)
    return chaves


def carregar_zoomout_config():
    """Configuração global do zoom out artificial."""
    cfg = carregar_json(ZOOMOUT_CONFIG_PATH, default={})
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault('zoom_out_pct', 0)
    return cfg


def salvar_zoomout_config(zoom_out_pct):
    """Salva o percentual global de zoom out."""
    salvar_json(ZOOMOUT_CONFIG_PATH, {
        'zoom_out_pct': int(zoom_out_pct),
        'atualizado_em': time.strftime('%Y-%m-%d %H:%M:%S')
    })


def obter_zoom_out_pct():
    """Retorna o percentual atual de zoom out, priorizando o estado do notebook."""
    state = globals().get('STATE', {})
    if isinstance(state, dict) and 'zoom_out_pct' in state:
        try:
            return int(state.get('zoom_out_pct', 0))
        except Exception:
            return 0
    return int(carregar_zoomout_config().get('zoom_out_pct', 0))


def _path_em_chaves(path, chaves):
    p = Path(str(path))
    return (str(path) in chaves) or (str(p) in chaves) or (p.name in chaves) or (p.stem in chaves)


def imagem_tem_zoomout_path(path, split=None):
    """Verifica se uma imagem original foi marcada para gerar versão com zoom out."""
    path = Path(str(path))
    # Evita aplicar zoom out duas vezes em cópias já geradas.
    if '__zoomout_' in path.stem:
        return False
    try:
        if ZOOMOUT_DATASET_DIR in path.parents:
            return False
    except Exception:
        pass

    splits = [split] if split in ['train', 'test'] else ['train', 'test']
    for sp in splits:
        if _path_em_chaves(path, carregar_chaves_zoomout(sp)):
            return True
    return False


def aplicar_zoom_out_imagem_e_bboxes(img_bgr, boxes=None, zoom_pct=0):
    """
    Aplica zoom out artificial mantendo o tamanho final da imagem.

    Exemplo: zoom_pct=30 reduz o conteúdo para 70% do tamanho original e
    preenche as margens resultantes com preto. As caixas em pixels são
    transformadas para a nova posição.
    """
    boxes = boxes or []
    pct = max(0, min(90, int(zoom_pct)))
    if pct <= 0:
        return img_bgr, list(boxes)

    h, w = img_bgr.shape[:2]
    scale = max(0.05, 1.0 - pct / 100.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(img_bgr)

    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_small

    boxes_zoom = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        nx1 = x0 + x1 * scale
        ny1 = y0 + y1 * scale
        nx2 = x0 + x2 * scale
        ny2 = y0 + y2 * scale
        boxes_zoom.append(clip_bbox(nx1, ny1, nx2, ny2, w, h))

    return canvas, boxes_zoom


def _zoomout_nome_arquivo(row, zoom_pct):
    p = Path(str(row['path']))
    return f"{p.stem}__zoomout_{int(zoom_pct):02d}{p.suffix.lower()}"


def caminhos_zoomout_para_row(row, zoom_pct):
    """Caminhos da cópia com zoom out aplicada para uma imagem do split."""
    split = row.get('split', 'train')
    nome_img = _zoomout_nome_arquivo(row, zoom_pct)
    stem = Path(nome_img).stem
    img_dir = ZOOMOUT_DATASET_DIR / split / 'images'
    label_dir = ZOOMOUT_DATASET_DIR / split / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return img_dir / nome_img, label_dir / f'{stem}.txt'


def escrever_bboxes_yolo(label_path, objetos, img_w, img_h):
    """Escreve bboxes em formato YOLO normalizado."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    linhas = []
    for obj in objetos:
        cls = int(obj.get('class_id', 0))
        x1, y1, x2, y2 = obj['bbox']
        x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, img_w, img_h)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 1 or bh < 1:
            continue
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        wn = bw / img_w
        hn = bh / img_h
        linhas.append(f"{cls} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
    label_path.write_text('\n'.join(linhas) + ('\n' if linhas else ''), encoding='utf-8')


def criar_imagem_zoomout_salva(row, zoom_pct=None, sobrescrever=False):
    """
    Cria uma cópia da imagem com zoom out aplicado e uma label YOLO ajustada.

    As imagens originais não são sobrescritas. A cópia é salva em
    `ZOOMOUT_DATASET_DIR/<split>/images` e a label correspondente em
    `ZOOMOUT_DATASET_DIR/<split>/labels`.
    """
    pct = int(obter_zoom_out_pct() if zoom_pct is None else zoom_pct)
    if pct <= 0:
        return None, None

    img_out_path, label_out_path = caminhos_zoomout_para_row(row, pct)
    if img_out_path.exists() and label_out_path.exists() and not sobrescrever:
        return img_out_path, label_out_path

    img = ler_imagem_bgr(row['path'])
    h, w = img.shape[:2]
    objetos_orig = ler_bboxes_yolo(row['label_path'], w, h)
    bboxes_orig = [obj['bbox'] for obj in objetos_orig]
    img_zoom, bboxes_zoom = aplicar_zoom_out_imagem_e_bboxes(img, bboxes_orig, pct)

    ok = cv2.imwrite(str(img_out_path), img_zoom)
    if not ok:
        raise RuntimeError(f'Falha ao salvar imagem com zoom out: {img_out_path}')

    objetos_zoom = []
    for obj, bbox_zoom in zip(objetos_orig, bboxes_zoom):
        objetos_zoom.append({'class_id': int(obj.get('class_id', 0)), 'bbox': bbox_zoom})
    escrever_bboxes_yolo(label_out_path, objetos_zoom, w, h)

    return img_out_path, label_out_path


def row_com_zoomout_salvo(row, zoom_pct=None):
    """Retorna uma cópia da linha apontando para a imagem/label com zoom out salva."""
    pct = int(obter_zoom_out_pct() if zoom_pct is None else zoom_pct)
    if pct <= 0:
        return row

    # Se a linha já aponta para uma cópia gerada, não gera outra.
    p_atual = Path(str(row['path']))
    if '__zoomout_' in p_atual.stem:
        return row

    img_out_path, label_out_path = criar_imagem_zoomout_salva(row, pct, sobrescrever=False)
    if img_out_path is None:
        return row

    new_row = row.copy()
    new_row['orig_path'] = str(row['path'])
    new_row['orig_label_path'] = str(row.get('label_path', ''))
    new_row['path'] = str(img_out_path)
    new_row['label_path'] = str(label_out_path)
    new_row['arquivo'] = img_out_path.name
    new_row['stem'] = img_out_path.stem
    new_row['zoom_out_aplicado_salvo'] = True
    new_row['zoom_out_pct'] = pct
    return new_row


def aplicar_zoomout_salvo_df(df_split, split):
    """Substitui linhas marcadas por linhas que apontam para cópias com zoom out já salvas."""
    if df_split is None or len(df_split) == 0:
        return df_split
    pct = int(obter_zoom_out_pct())
    if pct <= 0:
        return df_split.copy()

    chaves = carregar_chaves_zoomout(split)
    if not chaves:
        return df_split.copy()

    rows = []
    for _, row in df_split.iterrows():
        if imagem_tem_zoomout_path(row['path'], split):
            rows.append(row_com_zoomout_salvo(row, pct))
        else:
            rows.append(row.copy())
    return pd.DataFrame(rows).reset_index(drop=True)


def salvar_dataset_zoomout_para_selecoes(df=None, splits=('train', 'test'), zoom_pct=None, sobrescrever=False):
    """Gera/atualiza no disco as cópias com zoom out das imagens marcadas nos JSONs."""
    if df is None:
        df = globals().get('STATE', {}).get('df_dataset', pd.DataFrame())
    if df is None or len(df) == 0:
        return []

    pct = int(obter_zoom_out_pct() if zoom_pct is None else zoom_pct)
    gerados = []
    if pct <= 0:
        salvar_json(ZOOMOUT_MANIFEST_PATH, {'zoom_out_pct': pct, 'gerados': [], 'atualizado_em': time.strftime('%Y-%m-%d %H:%M:%S')})
        return gerados

    for split in splits:
        chaves = carregar_chaves_zoomout(split)
        if not chaves:
            continue
        df_split = df[df['split'] == split].copy()
        for _, row in df_split.iterrows():
            if imagem_tem_zoomout_path(row['path'], split):
                img_out, lab_out = criar_imagem_zoomout_salva(row, pct, sobrescrever=sobrescrever)
                if img_out is not None:
                    gerados.append({
                        'split': split,
                        'orig_path': str(row['path']),
                        'orig_label_path': str(row.get('label_path', '')),
                        'zoom_img_path': str(img_out),
                        'zoom_label_path': str(lab_out),
                        'zoom_out_pct': pct
                    })

    salvar_json(ZOOMOUT_MANIFEST_PATH, {
        'zoom_out_pct': pct,
        'n_gerados': len(gerados),
        'gerados': gerados,
        'atualizado_em': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    return gerados


def ler_imagem_row_com_zoomout(row, retornar_info=False):
    """Lê a imagem. Se a linha original foi marcada, usa a cópia salva com zoom out."""
    if imagem_tem_zoomout_path(row['path'], row.get('split')) and int(obter_zoom_out_pct()) > 0:
        row = row_com_zoomout_salvo(row)
    img = ler_imagem_bgr(row['path'])
    aplicar = bool(row.get('zoom_out_aplicado_salvo', False)) or ('__zoomout_' in Path(str(row['path'])).stem)
    info = {'zoom_out_aplicado': bool(aplicar), 'zoom_out_pct': int(row.get('zoom_out_pct', obter_zoom_out_pct() if aplicar else 0))}
    return (img, info) if retornar_info else img


def carregar_imagem_e_bboxes_row(row):
    """Lê imagem e bboxes YOLO. Se marcada, usa a cópia salva com zoom out e label já ajustada."""
    if imagem_tem_zoomout_path(row['path'], row.get('split')) and int(obter_zoom_out_pct()) > 0:
        row = row_com_zoomout_salvo(row)

    img = ler_imagem_bgr(row['path'])
    h, w = img.shape[:2]
    boxes_raw = ler_bboxes_yolo(row['label_path'], w, h)
    boxes = [b['bbox'] for b in boxes_raw]

    aplicar = bool(row.get('zoom_out_aplicado_salvo', False)) or ('__zoomout_' in Path(str(row['path'])).stem)
    return img, boxes, {'zoom_out_aplicado': bool(aplicar), 'zoom_out_pct': int(row.get('zoom_out_pct', obter_zoom_out_pct() if aplicar else 0))}


def aplicar_selecao_salva(df, split, usar_selecao=True):
    """Filtra o split usando a seleção manual persistente.

    Quando `usar_selecao=True`, os JSONs de seleção comandam quais imagens entram
    em treino/teste. Se o JSON ainda não existir, usa o split completo como fallback
    inicial. Se o JSON existir, mas nenhuma imagem casar com os nomes atuais, retorna
    vazio para deixar claro que a curadoria precisa ser revisada.
    """
    df_split = df[df['split'] == split].copy()
    if not usar_selecao:
        return aplicar_zoomout_salvo_df(df_split, split)

    sel_path = SELECAO_TREINO_PATH if split == 'train' else SELECAO_TESTE_PATH
    if not Path(sel_path).exists():
        return aplicar_zoomout_salvo_df(df_split, split)

    chaves = carregar_chaves_selecao(sel_path)
    if not chaves:
        return df_split.iloc[0:0].copy()

    mask = df_split.apply(
        lambda r: (str(r['path']) in chaves) or (r['arquivo'] in chaves) or (r['stem'] in chaves),
        axis=1
    )
    df_sel = df_split[mask].copy()
    return aplicar_zoomout_salvo_df(df_sel, split)


def limitar_df(df, max_n, random_state=42):
    max_n = int(max_n)
    if max_n <= 0 or len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=int(random_state)).copy()


def gerar_amostras_de_split(df_split, p, split_nome='train', status_callback=None):
    """
    Gera amostras positivas e negativas a partir de um split YOLO.

    Positivos: crops das bboxes anotadas.
    Negativos: crops aleatórios com IoU baixo em relação às bboxes.
    """
    rng = np.random.default_rng(int(p['random_state']) + (0 if split_nome == 'train' else 999))

    X = []
    y = []
    erros = []
    info = []

    neg_por_pos = int(p['negativos_por_positivo'])
    neg_iou_max = float(p['neg_iou_max'])
    pos_margin = float(p['pos_margin_pct'])
    neg_attempts = int(p['neg_attempts'])

    # Tamanho reserva para negativos em imagens sem objeto.
    wh_positivos = []

    total = len(df_split)
    for k, (_, row) in enumerate(df_split.iterrows(), start=1):
        if status_callback is not None and (k == 1 or k % 20 == 0 or k == total):
            status_callback(f'{split_nome}: gerando amostras {k}/{total}')

        try:
            img, boxes, zoom_info = carregar_imagem_e_bboxes_row(row)
            h, w = img.shape[:2]

            for bbox in boxes:
                bbox_pos = expandir_bbox(bbox, pos_margin, w, h)
                crop = recortar_bbox(img, bbox_pos)
                if crop.size == 0:
                    continue
                X.append(extrair_hog_de_imagem(crop, p))
                y.append(1)
                info.append({'split': split_nome, 'tipo': 'pos', 'arquivo': row['arquivo'], 'bbox': bbox_pos, **zoom_info})
                wh_positivos.append((bbox_pos[2] - bbox_pos[0], bbox_pos[3] - bbox_pos[1]))

            # Negativos proporcionais aos positivos da imagem.
            # Se a imagem não tiver objetos, gera pelo menos 1 negativo com tamanho mediano.
            n_base = max(1, len(boxes))
            n_neg = neg_por_pos * n_base

            if wh_positivos:
                median_w = int(np.median([v[0] for v in wh_positivos]))
                median_h = int(np.median([v[1] for v in wh_positivos]))
            else:
                median_w = max(24, int(w * 0.12))
                median_h = max(48, int(h * 0.25))

            for _ in range(n_neg):
                # Usa tamanho parecido com caixas reais quando possível.
                if boxes:
                    ref = boxes[int(rng.integers(0, len(boxes)))]
                    target_w = ref[2] - ref[0]
                    target_h = ref[3] - ref[1]
                else:
                    target_w = median_w
                    target_h = median_h

                # pequena variação de escala para aumentar diversidade.
                scale = float(rng.uniform(0.80, 1.25))
                tw = int(target_w * scale)
                th = int(target_h * scale)

                escolhido = None
                for _try in range(neg_attempts):
                    cand = bbox_aleatorio(w, h, tw, th, rng)
                    if not boxes or max(iou_bbox(cand, b) for b in boxes) <= neg_iou_max:
                        escolhido = cand
                        break

                if escolhido is None:
                    continue

                crop = recortar_bbox(img, escolhido)
                if crop.size == 0:
                    continue
                X.append(extrair_hog_de_imagem(crop, p))
                y.append(0)
                info.append({'split': split_nome, 'tipo': 'neg', 'arquivo': row['arquivo'], 'bbox': escolhido, **zoom_info})

        except Exception as e:
            erros.append((row.get('path', ''), str(e)))

    if len(X) == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int), erros, info

    return np.vstack(X), np.array(y, dtype=int), erros, info


def preparar_dados_treino_teste(df, p, status_callback=None):
    """Prepara X_train, X_test, y_train, y_test usando train/test reais quando possível."""
    usar_selecao = bool(p.get('usar_selecao_salva', True))
    rs = int(p['random_state'])

    df_train = aplicar_selecao_salva(df, 'train', usar_selecao=usar_selecao)
    df_test = aplicar_selecao_salva(df, 'test', usar_selecao=usar_selecao)

    df_train = limitar_df(df_train, int(p['max_train_images']), rs)
    df_test = limitar_df(df_test, int(p['max_test_images']), rs + 1)

    X_train_all, y_train_all, erros_train, info_train = gerar_amostras_de_split(
        df_train, p, split_nome='train', status_callback=status_callback
    )

    X_test_real, y_test_real, erros_test, info_test = gerar_amostras_de_split(
        df_test, p, split_nome='test', status_callback=status_callback
    )

    erros = erros_train + erros_test

    if len(y_train_all) < 4 or len(np.unique(y_train_all)) < 2:
        raise RuntimeError(
            'Amostras de treino insuficientes. Verifique se train/labels possui anotações YOLO válidas.'
        )

    usar_test_real = bool(p.get('usar_test_real', True))
    if usar_test_real and len(y_test_real) >= 2 and len(np.unique(y_test_real)) == 2:
        X_train, y_train = X_train_all, y_train_all
        X_test, y_test = X_test_real, y_test_real
        modo_eval = 'pasta_test'
    else:
        test_size = float(p['test_size'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_all, y_train_all,
            test_size=test_size,
            random_state=rs,
            stratify=y_train_all
        )
        modo_eval = 'holdout_train'

    stats = {
        'modo_eval': modo_eval,
        'n_img_train': int(len(df_train)),
        'n_img_test': int(len(df_test)),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'n_total': int(len(y_train) + len(y_test)),
        'n_pos_train': int((y_train == 1).sum()),
        'n_neg_train': int((y_train == 0).sum()),
        'n_pos_test': int((y_test == 1).sum()),
        'n_neg_test': int((y_test == 0).sum()),
        'n_pos_total': int((np.concatenate([y_train, y_test]) == 1).sum()),
        'n_neg_total': int((np.concatenate([y_train, y_test]) == 0).sum()),
        'n_erros': int(len(erros)),
    }

    return X_train, X_test, y_train, y_test, stats, erros, info_train + info_test


def assinatura_dataset(df):
    """Cria assinatura do dataset considerando imagens e arquivos de label."""
    if df is None or len(df) == 0:
        return 'dataset_vazio'

    rows = []
    for _, row in df.sort_values(['split', 'path']).iterrows():
        item = {
            'split': row.get('split'),
            'path': str(row.get('path')),
            'label_path': str(row.get('label_path')),
            'n_obj': int(row.get('n_obj', 0)),
        }
        for key in ['path', 'label_path']:
            pp = Path(item[key])
            if pp.exists():
                st = pp.stat()
                item[f'{key}_size'] = int(st.st_size)
                item[f'{key}_mtime'] = float(st.st_mtime)
            else:
                item[f'{key}_size'] = None
                item[f'{key}_mtime'] = None
        rows.append(item)

    # Também inclui os JSONs opcionais de seleção/zoom out, se existirem.
    for sel in [SELECAO_TREINO_PATH, SELECAO_TESTE_PATH, ZOOMOUT_TREINO_PATH, ZOOMOUT_TESTE_PATH, ZOOMOUT_CONFIG_PATH, ZOOMOUT_MANIFEST_PATH]:
        if sel.exists():
            st = sel.stat()
            rows.append({'selection_json': str(sel), 'size': int(st.st_size), 'mtime': float(st.st_mtime)})

    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def assinatura_treinamento(df, p):
    """Assinatura completa: dataset + parâmetros."""
    payload = {
        'versao_notebook': 'cilindros_hough_hog_svm_cylinders_yolo_v4_zoomout_salvo',
        'dataset': assinatura_dataset(df),
        'params': p,
        'zoom_out_pct': obter_zoom_out_pct()
    }
    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


def carregar_historico():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def salvar_linha_historico(row):
    hist = carregar_historico()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HISTORY_PATH, index=False)
    return hist


def abreviar_float(x, nd=4):
    try:
        return round(float(x), nd)
    except Exception:
        return x


# ============================================================
# 8A. Funções auxiliares de avaliação, NMS e métricas
# ============================================================
# Mantém somente o necessário para treinamento/validação.
# A antiga expansão X/Y e o antigo painel de refino não fazem parte desta versão.

def _pget(p, nome, default):
    return p[nome] if nome in p else default


def _bbox_valida(bbox):
    try:
        vals = [float(v) for v in bbox]
        return all(np.isfinite(v) for v in vals) and vals[2] > vals[0] and vals[3] > vals[1]
    except Exception:
        return False


def nms_bboxes_refinado(resultados_pos, iou_thr=0.30, max_det=8):
    if not resultados_pos:
        return []
    items = []
    for r in resultados_pos:
        if not _bbox_valida(r.get('bbox')):
            continue
        score = r.get('score_final_float', r.get('score_float', 0.0))
        if score is None:
            score = 0.0
        items.append((float(score), r))
    items.sort(key=lambda x: x[0], reverse=True)

    keep = []
    for _, r in items:
        if len(keep) >= int(max_det):
            break
        if all(iou_bbox(r['bbox'], k['bbox']) <= float(iou_thr) for k in keep):
            keep.append(r)
    return keep


def associar_predicoes_gt(pred_boxes, gt_boxes, iou_thr=0.50):
    """Associa predições a GT por IoU de forma gulosa."""
    pred_boxes = [tuple(map(float, b)) for b in pred_boxes if _bbox_valida(b)]
    gt_boxes = [tuple(map(float, b)) for b in gt_boxes if _bbox_valida(b)]
    pares = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            pares.append((iou_bbox(pb, gb), i, j))
    pares.sort(reverse=True, key=lambda x: x[0])

    usados_p = set()
    usados_g = set()
    ious_tp = []
    for iou, i, j in pares:
        if iou < float(iou_thr):
            break
        if i in usados_p or j in usados_g:
            continue
        usados_p.add(i)
        usados_g.add(j)
        ious_tp.append(float(iou))

    tp = len(ious_tp)
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'ious_tp': ious_tp, 'n_pred': len(pred_boxes), 'n_gt': len(gt_boxes)}


def _metricas_det_agregadas(contadores, iou_thr):
    tp = int(contadores.get('tp', 0)); fp = int(contadores.get('fp', 0)); fn = int(contadores.get('fn', 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ious = contadores.get('ious_tp', [])
    return {
        f'precision_det_{int(iou_thr*100)}': prec,
        f'recall_det_{int(iou_thr*100)}': rec,
        f'f1_det_{int(iou_thr*100)}': f1,
        f'tp_det_{int(iou_thr*100)}': tp,
        f'fp_det_{int(iou_thr*100)}': fp,
        f'fn_det_{int(iou_thr*100)}': fn,
        f'mean_iou_tp_{int(iou_thr*100)}': float(np.mean(ious)) if ious else 0.0,
    }


def avaliar_detector_completo(df, clf, p, status_callback=None):
    """Avalia o detector completo em imagens selecionadas de teste, comparando com YOLO por IoU."""
    if df is None or len(df) == 0 or clf is None:
        return {}

    usar_selecao = bool(_pget(p, 'usar_selecao_salva', True))
    df_eval = aplicar_selecao_salva(df, 'test', usar_selecao=usar_selecao)
    if df_eval is None or len(df_eval) == 0:
        df_eval = aplicar_selecao_salva(df, 'train', usar_selecao=usar_selecao)
    if df_eval is None or len(df_eval) == 0:
        return {}

    max_imgs = int(_pget(p, 'det_eval_max_images', 120))
    if max_imgs > 0 and len(df_eval) > max_imgs:
        df_eval = df_eval.sample(max_imgs, random_state=int(_pget(p, 'random_state', 42))).reset_index(drop=True)
    else:
        df_eval = df_eval.reset_index(drop=True)

    thr_main = float(_pget(p, 'det_iou_thr', 0.50))
    thrs = sorted(set([0.30, 0.50, round(thr_main, 2)]))
    cont = {thr: {'tp': 0, 'fp': 0, 'fn': 0, 'ious_tp': []} for thr in thrs}
    n_gt_total = 0
    n_pred_total = 0
    best_ious_all = []
    erros = 0

    for k, row in df_eval.iterrows():
        try:
            img_bgr, boxes_gt, _ = carregar_imagem_e_bboxes_row(row)
            det = detectar_candidatos_cilindro(
                img_bgr, clf=clf, p=p,
                score_min=float(_pget(p, 'det_score_min', 0.0)),
                nms_iou=float(_pget(p, 'det_nms_iou', 0.30)),
                max_det=int(_pget(p, 'det_max_det', 8)),
                aplicar_nms=True
            )
            pred_boxes = [r['bbox'] for r in det['positivos_finais'] if _bbox_valida(r.get('bbox'))]
            gt_boxes = [b for b in boxes_gt if _bbox_valida(b)]
            n_gt_total += len(gt_boxes)
            n_pred_total += len(pred_boxes)
            for pb in pred_boxes:
                best_ious_all.append(max([iou_bbox(pb, gb) for gb in gt_boxes], default=0.0))
            for thr in thrs:
                m = associar_predicoes_gt(pred_boxes, gt_boxes, iou_thr=thr)
                cont[thr]['tp'] += m['tp']; cont[thr]['fp'] += m['fp']; cont[thr]['fn'] += m['fn']; cont[thr]['ious_tp'].extend(m['ious_tp'])
        except Exception:
            erros += 1
        if status_callback is not None and (k + 1) % 25 == 0:
            status_callback(f'Avaliação detector completo: {k+1}/{len(df_eval)} imagens...')

    out = {
        'det_eval_imgs': int(len(df_eval)),
        'n_gt_det': int(n_gt_total),
        'n_pred_det': int(n_pred_total),
        'det_erros': int(erros),
        'det_iou_thr': float(thr_main),
        'mean_iou_best': float(np.mean(best_ious_all)) if best_ious_all else 0.0,
        'median_iou_best': float(np.median(best_ious_all)) if best_ious_all else 0.0,
    }
    for thr in thrs:
        out.update(_metricas_det_agregadas(cont[thr], thr))
    # aliases para a IoU principal escolhida no widget
    key = int(thr_main * 100)
    for prefix in ['precision_det', 'recall_det', 'f1_det', 'tp_det', 'fp_det', 'fn_det', 'mean_iou_tp']:
        out[prefix] = out.get(f'{prefix}_{key}', 0.0)
    return out


# ------------------------------------------------------------
# Visualizações atualizadas: ROI original, ROI refinada e retas encontradas
# ------------------------------------------------------------
COLOR_LINE_CANDIDATE = (0, 220, 255)  # ciano: retas usadas para formar a ROI candidata


# ============================================================
# 8B. Núcleo V2: ROIs somente por pares de retas paralelas
# ============================================================
# Estratégia desta versão:
# 1) detectar retas por Hough;
# 2) unir segmentos colineares e filtrar retas por geometria + suporte real;
# 3) formar ROIs somente com pares de retas paralelas;
# 4) calcular a BBox a partir do par de retas, com margem fixa em pixels;
# 5) pontuar cada ROI por overlap, similaridade angular e proximidade da razão L/W ao alvo ajustável.

# ------------------------------------------------------------
# Geometria básica de retas e caixas orientadas
# ------------------------------------------------------------
def _line_info(line):
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    L = float(math.hypot(dx, dy))
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L
    angle = (math.degrees(math.atan2(uy, ux)) + 180.0) % 180.0
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return {
        'line': tuple(map(int, [round(x1), round(y1), round(x2), round(y2)])),
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'length': L,
        'u': np.array([ux, uy], dtype=float),
        'v': np.array([-uy, ux], dtype=float),
        'center': np.array([cx, cy], dtype=float),
        'angle': angle
    }


def _angle_diff_180(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _interval_gap(a1, a2, b1, b2):
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def _projecoes_linha_em_base(line, origin, u, v):
    pts = np.array([[line[0], line[1]], [line[2], line[3]]], dtype=float)
    rel = pts - origin.reshape(1, 2)
    ts = rel @ u
    ss = rel @ v
    return float(np.min(ts)), float(np.max(ts)), float(np.mean(ss))


def _bbox_axis_from_points(pts, shape, margin=0):
    h, w = shape[:2]
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return None
    x1 = int(max(0, math.floor(np.min(pts[:, 0]) - margin)))
    y1 = int(max(0, math.floor(np.min(pts[:, 1]) - margin)))
    x2 = int(min(w, math.ceil(np.max(pts[:, 0]) + margin)))
    y2 = int(min(h, math.ceil(np.max(pts[:, 1]) + margin)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return (x1, y1, x2, y2)


def _oriented_corners(origin, u, v, tmin, tmax, smin, smax):
    pts = [
        origin + u * tmin + v * smin,
        origin + u * tmax + v * smin,
        origin + u * tmax + v * smax,
        origin + u * tmin + v * smax,
    ]
    return np.asarray(pts, dtype=np.float32)


def _bbox_center(bbox):
    if not _bbox_valida(bbox):
        return np.array([0.0, 0.0], dtype=float)
    x1, y1, x2, y2 = map(float, bbox)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def _bbox_area(bbox):
    if not _bbox_valida(bbox):
        return 0.0
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ------------------------------------------------------------
# União de segmentos e suporte real das retas
# ------------------------------------------------------------
def unir_segmentos_colineares(lines, p):
    """Une segmentos quase colineares antes de formar pares."""
    if not bool(_pget(p, 'merge_collinear_enabled', True)):
        return [tuple(map(int, l)) for l in lines]

    infos = [_line_info(l) for l in lines]
    infos = [i for i in infos if i is not None and i['length'] >= float(_pget(p, 'line_length_min', 0))]
    if not infos:
        return []

    gap_max = float(_pget(p, 'merge_gap_px', 18))
    angle_tol = float(_pget(p, 'merge_angle_tol_deg', 5.0))
    perp_tol = float(_pget(p, 'merge_perp_tol_px', 8.0))

    infos = sorted(infos, key=lambda d: d['length'], reverse=True)
    usados = set()
    merged = []

    for i, base in enumerate(infos):
        if i in usados:
            continue
        origin = base['center']
        u = base['u'].copy()
        v = base['v'].copy()
        t1, t2, _ = _projecoes_linha_em_base(base['line'], origin, u, v)
        intervalo_atual = [min(t1, t2), max(t1, t2)]
        grupo = [base]
        usados.add(i)

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(infos):
                if j in usados:
                    continue
                if _angle_diff_180(base['angle'], other['angle']) > angle_tol:
                    continue
                ot1, ot2, os = _projecoes_linha_em_base(other['line'], origin, u, v)
                omin, omax = min(ot1, ot2), max(ot1, ot2)
                if abs(os) > perp_tol:
                    continue
                if _interval_gap(intervalo_atual[0], intervalo_atual[1], omin, omax) > gap_max:
                    continue
                usados.add(j)
                grupo.append(other)
                intervalo_atual[0] = min(intervalo_atual[0], omin)
                intervalo_atual[1] = max(intervalo_atual[1], omax)
                changed = True

        t_vals, s_vals = [], []
        for g in grupo:
            gt1, gt2, gs = _projecoes_linha_em_base(g['line'], origin, u, v)
            t_vals.extend([gt1, gt2])
            s_vals.append(gs)
        tmin, tmax = float(np.min(t_vals)), float(np.max(t_vals))
        smed = float(np.mean(s_vals)) if s_vals else 0.0
        p1 = origin + u * tmin + v * smed
        p2 = origin + u * tmax + v * smed
        merged.append(tuple(map(int, [round(p1[0]), round(p1[1]), round(p2[0]), round(p2[1])])))

    return merged


def filtrar_linhas_remanescentes(lines, p):
    """Filtra linhas por comprimento. A orientação fica livre para captar cilindros inclinados."""
    out = []
    Lmin = float(_pget(p, 'line_length_min', 0))
    Lmax = float(_pget(p, 'line_length_max', 999999))
    for line in lines:
        info = _line_info(line)
        if info is None:
            continue
        if info['length'] < Lmin or info['length'] > Lmax:
            continue
        out.append(tuple(map(int, line)))
    return out


def _edge_has_support(edge_map, x, y, radius=2):
    if edge_map is None:
        return False
    h, w = edge_map.shape[:2]
    xi = int(round(x)); yi = int(round(y))
    r = int(max(0, radius))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    x1 = max(0, xi - r); x2 = min(w, xi + r + 1)
    y1 = max(0, yi - r); y2 = min(h, yi + r + 1)
    return bool(np.any(edge_map[y1:y2, x1:x2] > 0))


def _max_run_bool(vals):
    best = 0; cur = 0
    for v in vals:
        if bool(v):
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def avaliar_suporte_linha(line, pre, p):
    """Verifica se a reta tem pixels de borda acompanhando seu comprimento."""
    info = _line_info(line)
    if info is None:
        return {'ok': False, 'ratio': 0.0, 'max_gap_px': 9999.0}

    source = _pget(p, 'line_support_source', 'canny_hough')
    edge_map = pre.get(source, pre.get('canny_hough', pre.get('canny_full')))
    if edge_map is None:
        return {'ok': True, 'ratio': 1.0, 'max_gap_px': 0.0}

    radius = int(_pget(p, 'line_support_radius_px', 2))
    min_ratio = float(_pget(p, 'line_support_min_ratio', 0.30))
    max_gap_px = float(_pget(p, 'line_support_max_gap_px', 30))
    L = max(1.0, float(info['length']))
    step_px = 3.0
    n = int(np.clip(math.ceil(L / step_px) + 1, 8, 240))
    tvals = np.linspace(-L * 0.5, L * 0.5, n)
    pts = info['center'].reshape(1, 2) + tvals.reshape(-1, 1) * info['u'].reshape(1, 2)

    support = np.asarray([_edge_has_support(edge_map, x, y, radius=radius) for x, y in pts], dtype=bool)
    ratio = float(np.mean(support)) if len(support) else 0.0
    gap_px = float(_max_run_bool(~support) * step_px) if len(support) else 9999.0
    ok = (ratio >= min_ratio) and (gap_px <= max_gap_px or max_gap_px <= 0)
    return {'ok': bool(ok), 'ratio': ratio, 'max_gap_px': gap_px, 'n_samples': int(n)}


def filtrar_linhas_por_suporte(lines, pre, p):
    """Remove retas que não têm suporte real de borda."""
    if not bool(_pget(p, 'line_support_enabled', True)):
        return list(lines), {'n_linhas_suporte_testadas': int(len(lines)), 'n_linhas_suporte_rejeitadas': 0, 'line_support_debug': []}

    filtradas, debug = [], []
    for line in lines:
        info = avaliar_suporte_linha(line, pre, p)
        debug.append({'line': tuple(map(int, line)), **info})
        if info.get('ok', False):
            filtradas.append(tuple(map(int, line)))
    return filtradas, {
        'n_linhas_suporte_testadas': int(len(lines)),
        'n_linhas_suporte_rejeitadas': int(len(lines) - len(filtradas)),
        'line_support_debug': debug
    }


# ------------------------------------------------------------
# Pares paralelos: única fonte de ROIs
# ------------------------------------------------------------
def _score_lw_ratio(ratio, target, tol_pct):
    """Pontua a razão comprimento/largura da ROI em torno de um valor alvo."""
    ratio = float(ratio)
    target = max(1e-6, float(target))
    tol_pct = max(1e-6, float(tol_pct))
    erro_rel = abs(ratio - target) / target
    return float(max(0.0, 1.0 - min(1.0, erro_rel / tol_pct)))


def _normalizar_pesos_score(p):
    w_overlap = max(0.0, float(_pget(p, 'pair_score_overlap_weight', 0.45)))
    w_angle = max(0.0, float(_pget(p, 'pair_score_angle_weight', 0.30)))
    w_ratio = max(0.0, float(_pget(p, 'pair_score_ratio_weight', 0.25)))
    s = w_overlap + w_angle + w_ratio
    if s <= 1e-9:
        return 0.45, 0.30, 0.25
    return w_overlap / s, w_angle / s, w_ratio / s


def _score_par_v2(overlap, angle_diff, pair_tol, lw_ratio, p):
    """Score geométrico V2 baseado em overlap, paralelismo e razão comprimento/largura."""
    overlap_score = float(np.clip(overlap, 0.0, 1.0))
    angle_score = 1.0 - min(1.0, float(angle_diff) / max(1e-6, float(pair_tol)))
    ratio_score = _score_lw_ratio(
        lw_ratio,
        _pget(p, 'pair_lw_ratio_target', 5.0),
        _pget(p, 'pair_lw_ratio_tol_pct', 0.60),
    )
    w_overlap, w_angle, w_ratio = _normalizar_pesos_score(p)
    score = w_overlap * overlap_score + w_angle * angle_score + w_ratio * ratio_score
    return float(np.clip(score, 0.0, 1.0)), {
        'score_overlap': float(overlap_score),
        'score_angle': float(angle_score),
        'score_lw_ratio': float(ratio_score),
        'lw_ratio': float(lw_ratio),
        'lw_target': float(_pget(p, 'pair_lw_ratio_target', 5.0)),
    }


def _bbox_par_por_margem_fixa(origin, u, v, tmin, tmax, smin, smax, img_shape, p):
    """
    Calcula a BBox do par usando somente as extremidades das duas retas.

    Não há expansão por arcos, reta única ou proporção L/W. A razão L/W continua
    sendo usada apenas para pontuar a qualidade geométrica do par. A caixa final
    contém apenas o retângulo orientado definido pelo par de retas, acrescido de
    uma margem fixa em pixels (`roi_margin_px`).
    """
    width = max(1.0, float(smax - smin))
    base_len = max(1.0, float(tmax - tmin))
    margin_px = int(_pget(p, 'roi_margin_px', 10))

    # Retângulo orientado estritamente definido pelas extremidades do par.
    corners = _oriented_corners(
        origin,
        u,
        v,
        float(tmin),
        float(tmax),
        float(smin),
        float(smax)
    )

    # A margem é fixa em pixels e aplicada apenas ao converter para BBox no eixo da imagem.
    bbox = _bbox_axis_from_points(corners, img_shape, margin=margin_px)

    return bbox, corners, {
        'tmin': float(tmin),
        'tmax': float(tmax),
        'smin': float(smin),
        'smax': float(smax),
        'base_len': float(base_len),
        'width': float(width),
        'fixed_margin_px': int(margin_px),
        'lw_ratio_base': float(base_len / max(1.0, width)),
        'lw_ratio_bbox': float(base_len / max(1.0, width)),
    }


def gerar_pares_orientados(lines, img_shape, p, return_debug=False):
    """Gera ROIs iniciais somente a partir de pares de retas paralelas."""
    infos = [_line_info(l) for l in lines]
    infos = [i for i in infos if i is not None]
    candidatos = []
    dbg = {
        'n_linhas_para_pares': len(infos),
        'n_pares_testados': 0,
        'n_pares_ang_ok': 0,
        'n_pares_dist_ok': 0,
        'n_pares_overlap_ok': 0,
        'n_candidatos_par': 0,
    }

    pair_tol = float(_pget(p, 'pair_angle_tol_deg', 3.0))
    dist_min = float(_pget(p, 'pair_dist_min', 30))
    dist_max = float(_pget(p, 'pair_dist_max', 110))
    overlap_min = float(_pget(p, 'pair_overlap_min', 0.60))
    axis_gap_max = float(_pget(p, 'pair_axis_gap_px', 10))
    max_rois = int(_pget(p, 'max_rois', 50))
    pair_score_min = float(_pget(p, 'pair_score_min', 0.0))

    for i in range(len(infos)):
        a = infos[i]
        origin = a['center']
        u = a['u']; v = a['v']
        at1, at2, as0 = _projecoes_linha_em_base(a['line'], origin, u, v)
        amin, amax = min(at1, at2), max(at1, at2)

        for j in range(i + 1, len(infos)):
            dbg['n_pares_testados'] += 1
            b = infos[j]
            adiff = _angle_diff_180(a['angle'], b['angle'])
            if adiff > pair_tol:
                continue
            dbg['n_pares_ang_ok'] += 1

            bt1, bt2, bs0 = _projecoes_linha_em_base(b['line'], origin, u, v)
            bmin, bmax = min(bt1, bt2), max(bt1, bt2)

            dist = abs(bs0 - as0)
            if dist < dist_min or dist > dist_max:
                continue
            dbg['n_pares_dist_ok'] += 1

            ov = razao_sobreposicao((amin, amax), (bmin, bmax))
            gap_axis = _interval_gap(amin, amax, bmin, bmax)

            if ov < overlap_min and gap_axis > axis_gap_max:
                continue
            if ov >= overlap_min:
                dbg['n_pares_overlap_ok'] += 1

            tmin = min(amin, bmin)
            tmax = max(amax, bmax)
            smin, smax = sorted([as0, bs0])

            width_pair = max(1.0, smax - smin)
            base_len = max(1.0, tmax - tmin)
            lw_ratio = base_len / width_pair

            score, score_parts = _score_par_v2(ov, adiff, pair_tol, lw_ratio, p)
            if score < pair_score_min:
                continue

            bbox, corners, bbox_info = _bbox_par_por_margem_fixa(
                origin, u, v, tmin, tmax, smin, smax, img_shape, p
            )
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            roi_w = max(1, x2 - x1)
            roi_h = max(1, y2 - y1)
            aspect = roi_h / roi_w

            if not (float(_pget(p, 'roi_aspect_min', 0.3)) <= aspect <= float(_pget(p, 'roi_aspect_max', 15.0))):
                continue

            candidatos.append({
                'line1': a['line'], 'line2': b['line'],
                'line_base': a['line'], 'line_oposta': b['line'],
                'bbox': tuple(map(int, bbox)),
                'oriented_corners': corners,
                'origin': origin, 'u': u, 'v': v,
                'tmin': float(bbox_info['tmin']), 'tmax': float(bbox_info['tmax']),
                'smin': float(bbox_info['smin']), 'smax': float(bbox_info['smax']),
                'distance': float(dist),
                'overlap': float(ov),
                'gap_axis': float(gap_axis),
                'angle_diff': float(adiff),
                'angle': float(a['angle']),
                'area': int(roi_w * roi_h),
                'aspect': float(aspect),
                'lw_ratio': float(lw_ratio),
                'lw_ratio_bbox': float(bbox_info['lw_ratio_bbox']),
                'score_pair': float(score),
                'score_roi': float(score),
                'roi_score': float(score),
                'roi_evidence_score': float(score),
                'roi_evidence_class': '2_retas_paralelas',
                'tipo': 'par_retas_v2',
                **score_parts,
                **bbox_info,
            })

    candidatos = sorted(
        candidatos,
        key=lambda c: (c['score_pair'], c['overlap'], c['score_lw_ratio'], -c['angle_diff'], c['area']),
        reverse=True
    )[:max_rois]

    dbg['n_candidatos_par'] = len(candidatos)
    return (candidatos, dbg) if return_debug else candidatos


def expandir_candidato_orientado_por_score(img_bgr, pre, candidato, clf, p):
    """
    Compatibilidade com a pipeline anterior.

    Na V2, a BBox já é calculada com margem fixa na formação do par de retas.
    Portanto, esta função apenas reaproveita a BBox do candidato e, se houver
    classificador, calcula o score HOG/SVM.
    """
    bbox = candidato.get('bbox')
    if bbox is None or not _bbox_valida(bbox):
        return None, {'cap_score': 0.0}, {'refine_mode': 'bbox_invalida'}

    x1, y1, x2, y2 = map(int, bbox)
    crop = img_bgr[y1:y2, x1:x2].copy()
    pred = None
    svm_score = 0.0

    if clf is not None and crop.size > 0:
        try:
            feat = validar_features_modelo(extrair_hog_de_imagem(crop, p), clf, contexto="ROI V2")
            pred = int(clf.predict(feat)[0])
            try:
                svm_score = float(clf.decision_function(feat)[0])
            except Exception:
                svm_score = float(pred)
        except Exception:
            svm_score = -999.0

    geom_score = float(candidato.get('score_pair', 0.0))
    score_final = svm_score + 0.25 * geom_score if clf is not None else geom_score

    cap = {'cap_score': 0.0}
    ref_info = {
        'refine_mode': 'bbox_margem_fixa_v2',
        'tested': 1,
        'factor': 1.0,
        'score_final': float(score_final),
        'score_svm': svm_score,
        'pred_refine': pred,
        'geom_score': geom_score,
        'oriented_corners': candidato.get('oriented_corners'),
        'tmin': float(candidato.get('tmin', 0.0)),
        'tmax': float(candidato.get('tmax', 0.0)),
        'smin': float(candidato.get('smin', 0.0)),
        'smax': float(candidato.get('smax', 0.0)),
    }
    return tuple(map(int, bbox)), cap, ref_info


def _bbox_iou_e_overlap_menor(a, b):
    if not _bbox_valida(a) or not _bbox_valida(b):
        return 0.0, 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = _bbox_area(a); area_b = _bbox_area(b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    ov_small = inter / max(1e-6, min(area_a, area_b))
    return float(iou), float(ov_small)


def consolidar_rois_sobrepostas(candidatos, p):
    if not bool(_pget(p, 'roi_consolidation_enabled', True)):
        return list(candidatos), {
            'n_candidatos_pre_consolidacao': len(candidatos),
            'n_candidatos_pos_consolidacao': len(candidatos),
            'n_rois_consolidadas': 0
        }

    iou_thr = float(_pget(p, 'roi_consolidation_iou', 0.45))
    overlap_thr = float(_pget(p, 'roi_consolidation_overlap_small', 0.75))
    center_thr = float(_pget(p, 'roi_consolidation_center_px', 110))

    def rank(c):
        return (
            float(c.get('score_pair', 0.0)),
            float(c.get('overlap', 0.0)),
            float(c.get('score_lw_ratio', 0.0)),
            -float(c.get('angle_diff', 0.0)),
            float(c.get('area', 0.0)),
        )

    ordenados = sorted(candidatos, key=rank, reverse=True)
    usados = [False] * len(ordenados)
    saida = []
    removidas = 0

    for i, base in enumerate(ordenados):
        if usados[i]:
            continue
        usados[i] = True
        grupo = [base]
        cb = _bbox_center(base.get('bbox'))

        for j in range(i + 1, len(ordenados)):
            if usados[j]:
                continue
            cand = ordenados[j]
            iou, ov_small = _bbox_iou_e_overlap_menor(base.get('bbox'), cand.get('bbox'))
            dist_c = float(np.linalg.norm(cb - _bbox_center(cand.get('bbox'))))
            if (iou >= iou_thr) or (ov_small >= overlap_thr and (center_thr <= 0 or dist_c <= center_thr)):
                usados[j] = True
                grupo.append(cand)

        melhor = sorted(grupo, key=rank, reverse=True)[0].copy()
        melhor['n_rois_consolidadas'] = int(len(grupo))
        removidas += max(0, len(grupo) - 1)
        saida.append(melhor)

    return saida, {
        'n_candidatos_pre_consolidacao': len(candidatos),
        'n_candidatos_pos_consolidacao': len(saida),
        'n_rois_consolidadas': int(removidas)
    }


def candidatos_linhas_pares_v2(img_bgr, pre, p):
    fonte = _pget(p, 'hough_source_any', 'canny_hough')
    edges = pre.get(fonte, pre.get('canny_hough', pre['canny_full']))

    lines_raw = detectar_linhas_hough(edges, p)
    lines_merged = unir_segmentos_colineares(lines_raw, p)
    lines_geom = filtrar_linhas_remanescentes(lines_merged, p)
    lines_ok, debug_suporte = filtrar_linhas_por_suporte(lines_geom, pre, p)

    candidatos_brutos, debug_pares = gerar_pares_orientados(lines_ok, pre['gray'].shape, p, return_debug=True)
    candidatos, debug_consolidacao = consolidar_rois_sobrepostas(candidatos_brutos, p)
    candidatos = candidatos[:int(_pget(p, 'max_rois', 50))]

    info = {
        'lines_raw': lines_raw,
        'lines_merged': lines_merged,
        'lines_filtradas_geom': lines_geom,
        'lines_filtradas': lines_ok,
        'candidatos_sem_consolidacao': list(candidatos_brutos),
        'candidatos_consolidados': list(candidatos),
        'n_lines_raw': len(lines_raw),
        'n_lines_merged': len(lines_merged),
        'n_lines_filtradas_geom': len(lines_geom),
        'n_lines_filtradas': len(lines_ok),
        'n_candidatos_par': len(candidatos_brutos),
        'n_candidatos_brutos': len(candidatos_brutos),
        'n_candidatos_total': len(candidatos),
        'n_rois_pos_evidencia': len(candidatos_brutos),
    }
    info.update(debug_suporte)
    info.update(debug_pares)
    info.update(debug_consolidacao)
    return candidatos, info


def detectar_candidatos_cilindro(img_bgr, clf=None, p=None, score_min=None, nms_iou=None, max_det=None, aplicar_nms=True):
    if p is None:
        p = STATE['params']
    score_min = float(_pget(p, 'det_score_min', 0.0) if score_min is None else score_min)
    nms_iou = float(_pget(p, 'det_nms_iou', 0.30) if nms_iou is None else nms_iou)
    max_det = int(_pget(p, 'det_max_det', 8) if max_det is None else max_det)

    pre = aplicar_preprocessamento(img_bgr, p)
    candidatos, info_linhas = candidatos_linhas_pares_v2(img_bgr, pre, p)

    resultados, positivos = [], []
    erros = 0

    for idx, c in enumerate(candidatos, start=1):
        try:
            bbox_ref, cap, ref_info = expandir_candidato_orientado_por_score(img_bgr, pre, c, clf, p)
            if bbox_ref is None or not _bbox_valida(bbox_ref):
                continue

            x1, y1, x2, y2 = map(int, bbox_ref)
            crop = img_bgr[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            pred = ref_info.get('pred_refine')
            score_svm = ref_info.get('score_svm')
            score_final = ref_info.get('score_final')

            if clf is not None and pred is None:
                feat = validar_features_modelo(extrair_hog_de_imagem(crop, p), clf, contexto=f"ROI {idx}")
                pred = int(clf.predict(feat)[0])
                try:
                    score_svm = float(clf.decision_function(feat)[0])
                except Exception:
                    score_svm = float(pred)
                score_final = score_svm + 0.25 * float(c.get('score_pair', 0.0))

            r = {
                'roi': idx,
                'pred': 'cilindro' if pred == 1 else ('negativo' if pred == 0 else 'sem_modelo'),
                'bbox_original': c.get('bbox'),
                'bbox': tuple(map(int, bbox_ref)),
                'oriented_corners': ref_info.get('oriented_corners', c.get('oriented_corners')),
                'score': None if score_svm is None else round(float(score_svm), 3),
                'score_float': score_svm,
                'score_final': None if score_final is None else round(float(score_final), 3),
                'score_final_float': score_final,
                'score_geom': round(float(c.get('score_pair', 0.0)), 3),
                'score_pair': round(float(c.get('score_pair', 0.0)), 3),
                'score_roi': round(float(c.get('score_roi', c.get('score_pair', 0.0))), 3),
                'score_overlap': round(float(c.get('score_overlap', 0.0)), 3),
                'score_angle': round(float(c.get('score_angle', 0.0)), 3),
                'score_lw_ratio': round(float(c.get('score_lw_ratio', 0.0)), 3),
                'lw_ratio': round(float(c.get('lw_ratio', 0.0)), 2),
                'lw_target': round(float(c.get('lw_target', _pget(p, 'pair_lw_ratio_target', 5.0))), 2),
                'refine_mode': ref_info.get('refine_mode'),
                'dist_px': round(float(c.get('distance', 0.0)), 1),
                'score par': round(float(c.get('score_pair', 0.0)), 2),
                'score ROI': round(float(c.get('score_roi', c.get('score_pair', 0.0))), 3),
                'evidência ROI': c.get('roi_evidence_class', '2_retas_paralelas'),
                'overlap': round(float(c.get('overlap', 0.0)), 2),
                'ang_diff': round(float(c.get('angle_diff', 0.0)), 2),
                'angle': round(float(c.get('angle', 0.0)), 1),
                'tipo': c.get('tipo', 'par_retas_v2'),
            }

            resultados.append(r)

            if clf is not None and pred == 1 and (score_final is None or float(score_final) >= score_min):
                positivos.append(r)

        except Exception as e:
            erros += 1
            err_msg = str(e)[:240]
            print(f"[ERRO ROI {idx}] {err_msg}", flush=True)
            resultados.append({
                'roi': idx,
                'pred': 'erro',
                'bbox_original': c.get('bbox'),
                'bbox': c.get('bbox'),
                'erro': err_msg
            })

    positivos_finais = nms_bboxes_refinado(
        positivos,
        iou_thr=nms_iou,
        max_det=max_det
    ) if aplicar_nms else positivos

    ids_finais = {r['roi'] for r in positivos_finais}
    for r in resultados:
        r['nms_keep'] = bool(r.get('roi') in ids_finais)
        r['pred_visual'] = 'suprimido' if r.get('pred') == 'cilindro' and not r['nms_keep'] else r.get('pred')

    info = {
        'pred_raw': len(positivos),
        'pred_final': len(positivos_finais),
        'erro_rois': int(erros),
        'score_min': score_min,
        'nms_iou': nms_iou,
        'detector_strategy': 'pares_retas_v2'
    }
    info.update(info_linhas)
    return {
        'pre': pre,
        'candidatos': candidatos,
        'resultados': resultados,
        'positivos_finais': positivos_finais,
        'info': info
    }


# ------------------------------------------------------------
# Cores e visualizações
# ------------------------------------------------------------
COLOR_HOUGH_RAW = (0, 220, 255)
COLOR_HOUGH_FILTERED = (0, 220, 255)
COLOR_CANDIDATE_LINE = (0, 220, 255)
COLOR_ROI_INITIAL = (255, 140, 0)
COLOR_ROI_REFINED = (0, 255, 80)
COLOR_GT = (255, 220, 0)
COLOR_NEG = COLOR_ROI_INITIAL


def _desenhar_poligono_orientado(img_rgb, pts, color=(0, 255, 255), thickness=1):
    if pts is None:
        return img_rgb
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_rgb, [pts], True, color, thickness)
    return img_rgb


def _desenhar_retas_usadas_candidato(img_rgb, candidato, color=COLOR_CANDIDATE_LINE, thickness=1):
    """Desenha as retas usadas como evidência da detecção."""
    for line_key in ['line_base', 'line_oposta', 'line1', 'line2']:
        line = candidato.get(line_key)
        if line is None:
            continue
        x1, y1, x2, y2 = map(int, line)
        cv2.line(img_rgb, (x1, y1), (x2, y2), color, int(max(1, thickness)))


def _desenhar_numero_bbox(img_rgb, texto, x, y, color, font_scale=0.48, thickness=2):
    cv2.putText(
        img_rgb,
        str(texto),
        (int(x), int(max(12, y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        int(max(1, thickness)),
        cv2.LINE_AA
    )


def _desenhar_bbox_original_e_final(img_rgb, candidato, img_bgr_ref, pre, p, idx=None, line_thickness=1, bbox_thickness=1):
    """
    Desenha retas e BBoxes sem qualquer evidência curva.

    Convenção visual:
    - ciano: retas usadas como evidência;
    - laranja: BBox inicial/geométrica do par;
    - verde: BBox final usada no HOG/SVM.
    """
    _desenhar_retas_usadas_candidato(img_rgb, candidato, thickness=line_thickness)

    bbox_original = candidato.get('bbox')
    if _bbox_valida(bbox_original):
        x1, y1, x2, y2 = map(int, bbox_original)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), COLOR_ROI_INITIAL, int(max(1, bbox_thickness)))

    try:
        bbox_ref, cap, ref_info = expandir_candidato_orientado_por_score(img_bgr_ref, pre, candidato, None, p)
    except Exception:
        bbox_ref, cap, ref_info = None, {}, {}

    if bbox_ref is not None and _bbox_valida(bbox_ref):
        bx1, by1, bx2, by2 = map(int, bbox_ref)
        cv2.rectangle(img_rgb, (bx1, by1), (bx2, by2), COLOR_ROI_REFINED, int(max(1, bbox_thickness)))
        if idx is not None:
            score = float(candidato.get('score_pair', candidato.get('score_roi', 0.0)))
            _desenhar_numero_bbox(img_rgb, f'{idx} ({score:.2f})', bx1, by1 - 3, COLOR_ROI_REFINED)
    elif idx is not None and _bbox_valida(bbox_original):
        x1, y1, x2, y2 = map(int, bbox_original)
        _desenhar_numero_bbox(img_rgb, idx, x1, y1 - 3, COLOR_ROI_INITIAL)

    return img_rgb


def _desenhar_roi_inicial_candidato(img_rgb, candidato, idx=None):
    _desenhar_retas_usadas_candidato(img_rgb, candidato, thickness=1)
    if _bbox_valida(candidato.get('bbox')):
        x1, y1, x2, y2 = map(int, candidato['bbox'])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), COLOR_ROI_INITIAL, 1)
        if idx is not None:
            score = float(candidato.get('score_pair', candidato.get('score_roi', 0.0)))
            _desenhar_numero_bbox(img_rgb, f'{idx} ({score:.2f})', x1, y1 - 3, COLOR_ROI_INITIAL)


def _desenhar_roi_expandida_candidato(img_rgb, candidato, img_bgr_ref, pre, p, idx=None, line_thickness=1, bbox_thickness=1, **kwargs):
    return _desenhar_bbox_original_e_final(
        img_rgb,
        candidato,
        img_bgr_ref,
        pre,
        p,
        idx=idx,
        line_thickness=line_thickness,
        bbox_thickness=bbox_thickness
    )


def render_rois_iniciais_refinadas():
    pre = obter_preprocessamento_referencia(); img_bgr_ref = obter_imagem_referencia()
    if pre is None or img_bgr_ref is None:
        print('Nenhuma imagem de referência selecionada.'); return
    p = STATE['params']
    det = detectar_candidatos_cilindro(img_bgr_ref, clf=None, p=p, aplicar_nms=False)
    info = det['info']
    candidatos = list(info.get('candidatos_sem_consolidacao', det.get('candidatos', [])))[:int(_pget(p, 'max_rois', 50))]
    img = pre['rgb'].copy()
    for k, c in enumerate(candidatos, start=1):
        _desenhar_roi_inicial_candidato(img, c, idx=k)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=120)
    ax.imshow(img)
    ax.set_title(
        f"BBoxes por pares de retas antes do IoU | pares={info.get('n_candidatos_par', 0)} | "
        f"alvo L/W={_pget(p, 'pair_lw_ratio_target', 5.0)} | caixas={len(candidatos)}"
    )
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    plt.show()
    display(HTML('<small><b>Legenda:</b> ciano = retas usadas; laranja = BBox geométrica antes da consolidação.</small>'))
    if not candidatos:
        print('Nenhuma ROI inicial foi gerada. Ajuste Hough, suporte das retas ou critérios dos pares paralelos.')


def render_rois_expandidas_refinadas():
    pre = obter_preprocessamento_referencia(); img_bgr_ref = obter_imagem_referencia()
    if pre is None or img_bgr_ref is None:
        print('Nenhuma imagem de referência selecionada.'); return
    p = STATE['params']
    det = detectar_candidatos_cilindro(img_bgr_ref, clf=None, p=p, aplicar_nms=False)
    candidatos = det['candidatos']; info = det['info']
    atualizar_opcoes_roi_hog(candidatos)
    img = pre['rgb'].copy()
    for k, c in enumerate(candidatos[:int(_pget(p, 'max_rois', 50))], start=1):
        _desenhar_roi_expandida_candidato(img, c, img_bgr_ref, pre, p, idx=k)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=120)
    ax.imshow(img)
    ax.set_title(
        f"BBoxes com margem fixa + IoU | antes/depois="
        f"{info.get('n_candidatos_pre_consolidacao', len(candidatos))}/"
        f"{info.get('n_candidatos_pos_consolidacao', len(candidatos))} | finais={len(candidatos)}"
    )
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    plt.show()
    display(HTML('<small><b>Legenda:</b> ciano = retas usadas; laranja = BBox inicial; verde = BBox final após consolidação por IoU.</small>'))
    if not candidatos:
        print('Nenhuma ROI consolidada foi gerada.')


def render_rois_refinadas():
    render_rois_iniciais_refinadas()
    render_rois_expandidas_refinadas()


def _score_texto_candidato(c):
    try:
        return f"score={float(c.get('score_pair', c.get('score_roi', 0.0))):.2f}"
    except Exception:
        return "score=-"


def render_hog_visualizacao_refinada():
    pre = obter_preprocessamento_referencia(); img_bgr_ref = obter_imagem_referencia()
    if pre is None or img_bgr_ref is None:
        print('Nenhuma imagem de referência selecionada.'); return
    p = STATE['params']
    det = detectar_candidatos_cilindro(img_bgr_ref, clf=None, p=p, aplicar_nms=False)
    candidatos = det['candidatos']
    idx = atualizar_opcoes_roi_hog(candidatos)
    if idx is None or len(candidatos) == 0:
        print('Nenhuma ROI candidata foi gerada para esta imagem com os parâmetros atuais.'); return
    idx = int(np.clip(int(idx), 0, len(candidatos)-1))
    c = candidatos[idx]
    bbox_ref, cap, ref_info = expandir_candidato_orientado_por_score(img_bgr_ref, pre, c, None, p)
    if bbox_ref is None or not _bbox_valida(bbox_ref):
        print('A ROI selecionada gerou uma caixa inválida.'); return
    x1, y1, x2, y2 = map(int, bbox_ref)
    crop_bgr = img_bgr_ref[y1:y2, x1:x2].copy()
    if crop_bgr.size == 0:
        print('A ROI selecionada gerou um recorte vazio.'); return
    try:
        hog_input_resized, hog_image, features_hog = extrair_hog_visualizacao_de_imagem(crop_bgr, p)
        features_total = extrair_hog_de_imagem(crop_bgr, p)
    except Exception as e:
        print('Não foi possível gerar a visualização HOG:', e); traceback.print_exc(limit=1); return
    img_contexto = pre['rgb'].copy()
    _desenhar_roi_expandida_candidato(img_contexto, c, img_bgr_ref, pre, p, idx=idx+1)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    score_txt = _score_texto_candidato(c)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
    axs[0].imshow(img_contexto); axs[0].set_title(f'ROI/BBox selecionada\nROI {idx+1} | {score_txt}')
    axs[1].imshow(crop_rgb); axs[1].set_title('Recorte usado no SVM')
    axs[2].imshow(hog_input_resized, cmap='gray'); axs[2].set_title(f"Entrada HOG\n{p['hog_input']}")
    axs[3].imshow(hog_image, cmap='gray'); axs[3].set_title(f'HOG\nbase={len(features_hog)} | total={len(features_total)}')
    for ax in axs: ax.axis('off')
    plt.tight_layout(); plt.show()

    df_info = pd.DataFrame([{
        'roi': idx + 1,
        'score_roi': round(float(c.get('score_pair', c.get('score_roi', 0.0))), 3),
        'overlap': round(float(c.get('overlap', 0.0)), 2),
        'ang_diff': round(float(c.get('angle_diff', 0.0)), 2),
        'L/W': round(float(c.get('lw_ratio', 0.0)), 2),
        'alvo_L/W': round(float(c.get('lw_target', _pget(p, 'pair_lw_ratio_target', 5.0))), 2),
        'bbox': c.get('bbox'),
        'hog_input': p['hog_input'],
        'hog_size': f"{p['hog_resize_w']}x{p['hog_resize_h']}",
        'orient': p['hog_orientations'],
        'px_cell': p['hog_pixels_per_cell'],
        'cells_block': p['hog_cells_per_block'],
        'n_atributos': len(features_total)
    }])
    exibir_tabela_compacta(df_info, max_rows=1)


# ============================================================
# Runtime VS Code / Webcam
# ============================================================

def carregar_todos_setups():
    """Carrega setups oficiais e locais, sem exigir Jupyter."""
    setups = OrderedDict()

    def _adicionar(path, origem):
        data = carregar_json(path, default={})
        if not isinstance(data, dict):
            return
        raw = data.get('setups', {})
        if not isinstance(raw, dict):
            return

        for setup_id, setup in raw.items():
            if not isinstance(setup, dict):
                continue
            params = setup.get('params', {})
            if not isinstance(params, dict):
                continue
            nome = (
                setup.get('nome')
                or setup.get('label')
                or setup.get('descricao')
                or setup_id
            )
            key = str(setup_id)
            if key in setups:
                key = f"{key} [{origem}]"
            setups[key] = {
                'id': str(setup_id),
                'key': key,
                'nome': str(nome),
                'origem': origem,
                'path': str(path),
                'params': OrderedDict(params),
            }

    _adicionar(OFFICIAL_PARAM_SETUPS_PATH, 'official')
    _adicionar(USER_PARAM_SETUPS_PATH, 'user')
    return setups


def carregar_params_metadata():
    """Carrega os parâmetros usados no último treinamento, se existirem."""
    meta = carregar_json(METADATA_PATH, default={})
    if isinstance(meta, dict) and isinstance(meta.get('params'), dict):
        return OrderedDict(meta['params']), meta
    return None, meta if isinstance(meta, dict) else {}


def listar_setups_console(setups, meta=None):
    print("\nSetups disponíveis:")
    if not setups:
        print("  Nenhum setup JSON encontrado.")
    else:
        for i, (key, s) in enumerate(setups.items(), start=1):
            print(f"  {i:02d}. {key} | {s['nome']} | origem={s['origem']}")

    if isinstance(meta, dict) and isinstance(meta.get('params'), dict):
        print("\nTambém disponível:")
        print(f"  metadata | parâmetros do último treinamento | setup={meta.get('setup_id', '-')}, nome={meta.get('setup_nome', '-')}")
    print("")


def escolher_params(setup_arg=None, preferir_metadata=False):
    setups = carregar_todos_setups()
    params_meta, meta = carregar_params_metadata()

    if preferir_metadata:
        if params_meta is None:
            raise RuntimeError("Metadata do modelo não possui parâmetros salvos.")
        return params_meta, f"metadata/{meta.get('setup_id', 'sem_id')}"

    if setup_arg:
        if setup_arg.lower() in ['metadata', 'treinado', 'trained']:
            if params_meta is None:
                raise RuntimeError("Metadata do modelo não possui parâmetros salvos.")
            return params_meta, f"metadata/{meta.get('setup_id', 'sem_id')}"

        # Aceita chave, id ou nome.
        setup_arg_norm = str(setup_arg).strip().lower()
        for key, s in setups.items():
            if setup_arg_norm in [
                str(key).lower(),
                str(s['id']).lower(),
                str(s['nome']).lower(),
            ]:
                return s['params'], f"{s['id']} ({s['origem']})"

        raise RuntimeError(f"Setup não encontrado: {setup_arg}")

    # Sem argumento: tenta perguntar no terminal.
    listar_setups_console(setups, meta=meta)
    if not setups and params_meta is None:
        raise RuntimeError("Nenhum setup JSON ou metadata de treinamento foi encontrado.")

    if setups:
        resposta = input("Escolha o número do setup, ou digite 'm' para metadata do treinamento: ").strip()
        if resposta.lower() in ['m', 'meta', 'metadata']:
            if params_meta is None:
                raise RuntimeError("Metadata do modelo não possui parâmetros salvos.")
            return params_meta, f"metadata/{meta.get('setup_id', 'sem_id')}"

        try:
            idx = int(resposta)
            key = list(setups.keys())[idx - 1]
            s = setups[key]
            return s['params'], f"{s['id']} ({s['origem']})"
        except Exception:
            print("Entrada inválida. Usando o primeiro setup da lista.")
            key = next(iter(setups.keys()))
            s = setups[key]
            return s['params'], f"{s['id']} ({s['origem']})"

    return params_meta, f"metadata/{meta.get('setup_id', 'sem_id')}"


def carregar_modelo(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo HOG/SVM não encontrado em:\n{model_path}\n\n"
            "Treine o modelo no notebook Jupyter antes de rodar a webcam."
        )
    return joblib.load(model_path)


def desenhar_apenas_bboxes(frame_bgr, deteccoes, mostrar_score=True):
    """Desenha somente a BBox final da detecção, sem retas e sem curvas."""
    out = frame_bgr.copy()
    for r in deteccoes:
        bbox = r.get('bbox')
        if not _bbox_valida(bbox):
            continue
        x1, y1, x2, y2 = map(int, bbox)

        # OpenCV usa BGR: verde = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if mostrar_score:
            score = r.get('score_final')
            if score is None:
                score = r.get('score')
            texto = "cilindro"
            if score is not None:
                try:
                    texto += f" {float(score):.2f}"
                except Exception:
                    pass

            cv2.putText(
                out,
                texto,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
    return out




RESOLUTION_PRESETS = [
    (320, 240),
    (424, 240),
    (640, 480),
    (800, 600),
    (960, 540),
    (1280, 720),
]


def _indice_resolucao_inicial(width=None, height=None):
    """Escolhe o índice do preset mais próximo da resolução inicial."""
    if width is None or height is None:
        return 0
    alvo = (int(width), int(height))
    dist = [abs(w - alvo[0]) + abs(h - alvo[1]) for w, h in RESOLUTION_PRESETS]
    return int(np.argmin(dist))


def _score_para_slider(score_min):
    """Mapeia score_min [-30.0, +10.0] para slider inteiro [0, 400]."""
    try:
        score = float(score_min)
    except Exception:
        score = 0.0
    score = max(-30.0, min(10.0, score))
    return int(round((score + 30.0) * 10.0))


def _slider_para_score(v):
    """Mapeia slider inteiro [0, 400] para score_min [-30.0, +10.0]."""
    return float(v) / 10.0 - 30.0


class PainelControlesRuntime:
    """Painel visual próprio, sem trackbars nativas do OpenCV.

    A janela é desenhada como uma imagem normal, com sliders, legendas e valores.
    Isso evita a área preta e a falta de textos que podem ocorrer com as trackbars
    nativas do backend Qt do OpenCV em alguns ambientes Linux.
    """

    def __init__(self, score_min=0.0, detect_every=5, width=None, height=None):
        self.win = "BLAZE - controles"
        self.canvas_w = 620
        self.canvas_h = 250
        self.x0 = 225
        self.x1 = 545
        self.rows = {
            'res_idx': 78,
            'detect_every': 135,
            'score_slider': 192,
        }
        self.drag_key = None

        self.res_idx = int(_indice_resolucao_inicial(width, height))
        self.detect_every = int(max(1, min(30, int(detect_every))))
        self.score_slider = int(_score_para_slider(score_min))

        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self.desenhar()

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _valor_para_x(self, valor, vmin, vmax):
        if vmax <= vmin:
            return self.x0
        t = (float(valor) - float(vmin)) / float(vmax - vmin)
        return int(round(self.x0 + self._clamp(t, 0.0, 1.0) * (self.x1 - self.x0)))

    def _x_para_valor(self, x, vmin, vmax, inteiro=True):
        t = (float(x) - self.x0) / float(self.x1 - self.x0)
        v = float(vmin) + self._clamp(t, 0.0, 1.0) * float(vmax - vmin)
        return int(round(v)) if inteiro else v

    def _key_proxima(self, y):
        return min(self.rows.keys(), key=lambda k: abs(int(y) - self.rows[k]))

    def _atualizar_por_mouse(self, key, x):
        if key == 'res_idx':
            self.res_idx = self._clamp(self._x_para_valor(x, 0, len(RESOLUTION_PRESETS) - 1), 0, len(RESOLUTION_PRESETS) - 1)
        elif key == 'detect_every':
            self.detect_every = self._clamp(self._x_para_valor(x, 1, 30), 1, 30)
        elif key == 'score_slider':
            self.score_slider = self._clamp(self._x_para_valor(x, 0, 400), 0, 400)
        self.desenhar()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            key = self._key_proxima(y)
            if abs(y - self.rows[key]) <= 24:
                self.drag_key = key
                self._atualizar_por_mouse(key, x)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_key is not None:
            self._atualizar_por_mouse(self.drag_key, x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_key is not None:
                self._atualizar_por_mouse(self.drag_key, x)
            self.drag_key = None

    def _desenhar_slider(self, img, key, label, value_text, vmin, vmax, valor, ajuda):
        y = self.rows[key]
        cv2.putText(img, label, (24, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, value_text, (24, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (70, 70, 70), 1, cv2.LINE_AA)

        cv2.line(img, (self.x0, y), (self.x1, y), (175, 175, 175), 8, cv2.LINE_AA)
        cv2.line(img, (self.x0, y), (self._valor_para_x(valor, vmin, vmax), y), (45, 125, 220), 8, cv2.LINE_AA)

        # Marcas discretas.
        n_marks = 6 if key != 'res_idx' else len(RESOLUTION_PRESETS)
        for i in range(n_marks):
            if n_marks <= 1:
                x = self.x0
            else:
                x = int(round(self.x0 + i * (self.x1 - self.x0) / (n_marks - 1)))
            cv2.line(img, (x, y + 12), (x, y + 17), (120, 120, 120), 1, cv2.LINE_AA)

        xk = self._valor_para_x(valor, vmin, vmax)
        cv2.circle(img, (xk, y), 12, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (xk, y), 12, (45, 125, 220), 2, cv2.LINE_AA)
        cv2.putText(img, ajuda, (self.x0, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (95, 95, 95), 1, cv2.LINE_AA)

    def desenhar(self):
        img = np.full((self.canvas_h, self.canvas_w, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (self.canvas_w - 1, self.canvas_h - 1), (210, 210, 210), 1)
        cv2.putText(img, "BLAZE - ajustes da webcam", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "Arraste os marcadores azuis. Feche com q na janela da webcam.", (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1, cv2.LINE_AA)

        width, height = RESOLUTION_PRESETS[self.res_idx]
        self._desenhar_slider(
            img, 'res_idx', "Resolucao", f"{width} x {height}",
            0, len(RESOLUTION_PRESETS) - 1, self.res_idx,
            "maior = mais detalhe, mas deteccao mais pesada",
        )
        self._desenhar_slider(
            img, 'detect_every', "Atualizacao da deteccao", f"a cada {self.detect_every} frame(s)",
            1, 30, self.detect_every,
            "menor = atualiza mais rapido; maior = mais fluido",
        )
        score_min = _slider_para_score(self.score_slider)
        self._desenhar_slider(
            img, 'score_slider', "Score minimo", f"{score_min:.1f}",
            0, 400, self.score_slider,
            "mais negativo = mais permissivo; mais positivo = mais rigoroso",
        )
        cv2.imshow(self.win, img)

    def ler(self):
        self.desenhar()
        width, height = RESOLUTION_PRESETS[self.res_idx]
        return {
            'res_idx': int(self.res_idx),
            'width': int(width),
            'height': int(height),
            'detect_every': int(self.detect_every),
            'score_min': float(_slider_para_score(self.score_slider)),
        }


def criar_controles_runtime(score_min=0.0, detect_every=5, width=None, height=None):
    """Cria painel de controles desenhado manualmente com OpenCV."""
    return PainelControlesRuntime(score_min=score_min, detect_every=detect_every, width=width, height=height)


def ler_controles_runtime(painel):
    """Lê controles do painel customizado."""
    try:
        return painel.ler()
    except Exception:
        return None


def abrir_camera(camera_index=0, width=None, height=None):
    """Abre webcam com fallback para Windows/VS Code."""
    backends = []
    if os.name == 'nt':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    else:
        backends = [0]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(int(camera_index), backend) if backend else cv2.VideoCapture(int(camera_index))
            if cap is not None and cap.isOpened():
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass

    raise RuntimeError(f"Não foi possível abrir a webcam índice {camera_index}.")


def _executar_deteccao_runtime(frame, params, clf, score_min=0.0, nms_iou=0.30, max_det=8):
    """Executa a pipeline pesada de detecção em um frame.

    Esta função é separada para permitir execução assíncrona: a janela da webcam
    continua atualizando enquanto a detecção clássica roda em segundo plano.
    """
    det = detectar_candidatos_cilindro(
        frame,
        clf=clf,
        p=params,
        score_min=score_min,
        nms_iou=nms_iou,
        max_det=max_det,
        aplicar_nms=True
    )
    return det.get('positivos_finais', []), det.get('info', {})


def rodar_webcam(params, clf, camera_index=0, score_min=0.0, nms_iou=0.30, max_det=8,
                 width=None, height=None, espelhar=False, resize_display=1.0,
                 detect_every=5, max_rois_runtime=None, async_detection=True,
                 controles=True):
    """Executa webcam com inferência clássica.

    Para evitar que a janela pareça travada, a detecção pode rodar de forma
    assíncrona. Assim, a captura e exibição seguem em tempo real e a BBox mais
    recente é reaproveitada até a próxima detecção ficar pronta.
    """
    params_runtime = dict(params)
    if max_rois_runtime is not None:
        params_runtime['max_rois'] = int(max(1, max_rois_runtime))

    detect_every = int(max(1, detect_every))
    cap = abrir_camera(camera_index=camera_index, width=width, height=height)

    controles_win = None
    controles_estado = None
    res_idx_atual = _indice_resolucao_inicial(width, height)
    score_min_runtime = float(score_min)
    detect_every_runtime = int(detect_every)

    if controles:
        controles_win = criar_controles_runtime(
            score_min=score_min_runtime,
            detect_every=detect_every_runtime,
            width=width,
            height=height,
        )
        controles_estado = ler_controles_runtime(controles_win)
        if controles_estado is not None:
            res_idx_atual = controles_estado['res_idx']
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, controles_estado['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, controles_estado['height'])
            score_min_runtime = controles_estado['score_min']
            detect_every_runtime = controles_estado['detect_every']

    print("\nWebcam iniciada.")
    print("Pressione 'q' para sair.")
    print("A tela mostra apenas a BBox final da detecção.")
    print(f"Modo assíncrono: {bool(async_detection)} | detect_every inicial={detect_every_runtime} | max_rois_runtime={params_runtime.get('max_rois')}")
    if controles:
        print("Controles ativos: resolucao | detect_every | score_min_x10.")
        print("score_min_x10: -50 significa score_min=-5.0; -200 significa score_min=-20.0.\n")
    else:
        print()

    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0
    det_fps_t0 = time.time()
    det_count = 0
    det_fps = 0.0
    frame_idx = 0

    ultimas_deteccoes = []
    ultimo_info = {}
    ultimo_erro = None
    last_submit_time = 0.0
    future = None

    executor = ThreadPoolExecutor(max_workers=1) if async_detection else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Falha ao capturar frame da webcam.")
                break

            if espelhar:
                frame = cv2.flip(frame, 1)

            frame_idx += 1

            if controles and controles_win is not None:
                controles_estado = ler_controles_runtime(controles_win)
                if controles_estado is not None:
                    score_min_runtime = controles_estado['score_min']
                    detect_every_runtime = controles_estado['detect_every']
                    if controles_estado['res_idx'] != res_idx_atual:
                        res_idx_atual = controles_estado['res_idx']
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, controles_estado['width'])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, controles_estado['height'])
                        ultimas_deteccoes = []
                        ultimo_info = {}
                        print(
                            f"[CONTROLE] resolucao={controles_estado['width']}x{controles_estado['height']} | "
                            f"detect_every={detect_every_runtime} | score_min={score_min_runtime:.1f}",
                            flush=True,
                        )

            should_detect = (frame_idx % detect_every_runtime == 1)

            if async_detection:
                # Coleta resultado pronto, se houver.
                if future is not None and future.done():
                    try:
                        ultimas_deteccoes, ultimo_info = future.result()
                        ultimo_erro = None
                        det_count += 1
                    except Exception as e:
                        ultimas_deteccoes = []
                        ultimo_info = {}
                        ultimo_erro = str(e)[:120]
                    future = None

                # Submete nova detecção apenas se não há outra rodando.
                if should_detect and future is None:
                    frame_para_detectar = frame.copy()
                    future = executor.submit(
                        _executar_deteccao_runtime,
                        frame_para_detectar,
                        params_runtime,
                        clf,
                        score_min_runtime,
                        nms_iou,
                        max_det
                    )
                    last_submit_time = time.time()
            else:
                # Modo síncrono: mais simples, porém a janela pode travar durante a detecção.
                if should_detect:
                    try:
                        ultimas_deteccoes, ultimo_info = _executar_deteccao_runtime(
                            frame,
                            params_runtime,
                            clf,
                            score_min_runtime,
                            nms_iou,
                            max_det
                        )
                        ultimo_erro = None
                        det_count += 1
                    except Exception as e:
                        ultimas_deteccoes = []
                        ultimo_info = {}
                        ultimo_erro = str(e)[:120]

            saida = desenhar_apenas_bboxes(frame, ultimas_deteccoes, mostrar_score=True)
            n_det = len(ultimas_deteccoes)

            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / max(1e-6, now - fps_t0)
                fps_t0 = now
                fps_count = 0
            if now - det_fps_t0 >= 1.0:
                det_fps = det_count / max(1e-6, now - det_fps_t0)
                det_fps_t0 = now
                det_count = 0

            status = "detectando..." if (async_detection and future is not None) else "det ok"
            cv2.putText(saida, f"view FPS {fps:.1f} | det FPS {det_fps:.1f} | det {n_det} | {status}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(saida, f"score_min {score_min_runtime:.1f} | detect_every {detect_every_runtime} | res idx {res_idx_atual}", (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

            if ultimo_erro:
                cv2.putText(saida, f"erro: {ultimo_erro}", (12, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

            if resize_display and abs(float(resize_display) - 1.0) > 1e-6:
                saida = cv2.resize(saida, None, fx=float(resize_display), fy=float(resize_display),
                                   interpolation=cv2.INTER_AREA)

            cv2.imshow("BLAZE - detector classico de cilindros", saida)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Runtime VS Code para webcam do detector clássico HOG/SVM de cilindros."
    )
    parser.add_argument('--camera', type=int, default=0, help='Índice da webcam. Padrão: 0.')
    parser.add_argument('--setup', type=str, default=None, help='ID/nome do setup JSON. Use "metadata" para parâmetros do último treino.')
    parser.add_argument('--trained-params', action='store_true', help='Usa diretamente os parâmetros salvos no metadata do último treinamento.')
    parser.add_argument('--list-setups', action='store_true', help='Lista setups disponíveis e encerra.')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH), help='Caminho do modelo .joblib treinado.')
    parser.add_argument('--score-min', type=float, default=0.0, help='Score mínimo final para manter detecção.')
    parser.add_argument('--nms-iou', type=float, default=0.30, help='IoU do NMS.')
    parser.add_argument('--max-det', type=int, default=8, help='Máximo de detecções por frame.')
    parser.add_argument('--width', type=int, default=None, help='Largura desejada da captura.')
    parser.add_argument('--height', type=int, default=None, help='Altura desejada da captura.')
    parser.add_argument('--mirror', action='store_true', help='Espelha a webcam horizontalmente.')
    parser.add_argument('--display-scale', type=float, default=1.0, help='Escala de exibição da janela.')
    parser.add_argument('--detect-every', type=int, default=5, help='Executa a detecção pesada a cada N frames. Padrão: 5.')
    parser.add_argument('--max-rois-runtime', type=int, default=None, help='Limita max_rois durante a webcam para melhorar desempenho.')
    parser.add_argument('--sync-detection', action='store_true', help='Desativa detecção assíncrona. Útil apenas para depuração.')
    parser.add_argument('--no-controls', action='store_true', help='Desativa a janela de controles com sliders OpenCV.')

    args = parser.parse_args()

    setups = carregar_todos_setups()
    _, meta = carregar_params_metadata()

    if args.list_setups:
        listar_setups_console(setups, meta=meta)
        return

    params, setup_desc = escolher_params(args.setup, preferir_metadata=args.trained_params)
    clf = carregar_modelo(args.model)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Modelo:", args.model)
    print("Setup usado:", setup_desc)

    rodar_webcam(
        params=params,
        clf=clf,
        camera_index=args.camera,
        score_min=args.score_min,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        width=args.width,
        height=args.height,
        espelhar=args.mirror,
        resize_display=args.display_scale,
        detect_every=args.detect_every,
        max_rois_runtime=args.max_rois_runtime,
        async_detection=not args.sync_detection,
        controles=not args.no_controls,
    )


if __name__ == '__main__':
    main()
