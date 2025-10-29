# neural/evaluate.py
from core.geometry import random_triangles, triangle_centroids
import torch

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from core.bvh import build_bvh, collect_leaves
from core.metrics import compute_epo, compute_sah_for_split

def build_bvh_with_model(tris, model, max_leaf_size=8):
    """
    Constrói uma BVH simples guiada pelo modelo neural.
    Retorna um dicionário com estrutura hierárquica.
    """

    device = next(model.parameters()).device

    def recursive(indices):
        # Caso base: nó folha
        if len(indices) == 0:
            return {"leaf": []}
        if len(indices) <= max_leaf_size:
            return {"leaf": indices}

        # Subconjunto de triângulos e centróides
        tris_subset = tris[indices]
        cents = np.mean(tris_subset, axis=1)

        # Extrai features (média e desvio padrão)
        feat = np.concatenate([cents.mean(axis=0), cents.std(axis=0)]).astype(np.float32)

        # Move para o mesmo device do modelo
        with torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            pred = model(x)
            pos = float(pred.cpu().numpy()[0])  # volta p/ CPU só no fim

        # Define eixo e posição de corte
        axis = int(np.argmax(cents.std(axis=0)))
        mins, maxs = cents[:, axis].min(), cents[:, axis].max()
        split_pos = mins + pos * (maxs - mins)

        # Divide índices
        left_idx = [i for i in indices if np.mean(tris[i], axis=0)[axis] <= split_pos]
        right_idx = [i for i in indices if i not in left_idx]

        # Garante que não haja partição vazia
        if len(left_idx) == 0 or len(right_idx) == 0:
            return {"leaf": indices}

        # Recursão
        left_node = recursive(left_idx)
        right_node = recursive(right_idx)

        return {
            "axis": axis,
            "pos": split_pos,
            "left": left_node,
            "right": right_node,
            "is_leaf": False
        }

    # Chama a recursão a partir de todos os triângulos
    root = recursive(list(range(len(tris))))
    return root

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from core.bvh import build_bvh, collect_leaves
from core.metrics import compute_epo, compute_sah_for_split
from neural.evaluate import build_bvh_with_model


def compare(tris, model):
    """
    Compara BVH tradicional (SAH) e BVH neural (EPO),
    gera gráfico combinado com normalização e salva PNG/SVG.
    """
    print(f"🧩 Avaliando BVH neural no dispositivo: {next(model.parameters()).device}")

    # ==============================================================
    # 1️⃣ Gera BVH tradicional e calcula métricas
    # ==============================================================
    bvh_std = build_bvh(tris, max_leaf_size=8)
    leaves_std = collect_leaves(bvh_std)
    epo_std = compute_epo(tris, leaves_std)

    # Calcula SAH médio aproximado (amostragem simplificada)
    sah_samples_std = []
    for axis in range(3):
        cost, _, _ = compute_sah_for_split(list(range(len(tris))), tris, axis, 0.5)
        sah_samples_std.append(cost)
    sah_std = float(np.mean(sah_samples_std))

    # ==============================================================
    # 2️⃣ Gera BVH neural e calcula métricas
    # ==============================================================
    bvh_neural = build_bvh_with_model(tris, model, max_leaf_size=8)
    leaves_neu = collect_leaves(bvh_neural)
    epo_neu = compute_epo(tris, leaves_neu)

    sah_samples_neu = []
    for axis in range(3):
        cost, _, _ = compute_sah_for_split(list(range(len(tris))), tris, axis, 0.5)
        sah_samples_neu.append(cost)
    sah_neu = float(np.mean(sah_samples_neu))

    # ==============================================================
    # 3️⃣ Salva resultados em texto
    # ==============================================================
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    out_txt = os.path.join(results_dir, "compare_summary.txt")
    out_png = os.path.join(results_dir, "compare_plot.png")
    out_svg = os.path.join(results_dir, "compare_plot.svg")

    res = {
        "EPO": {"std": epo_std, "neu": epo_neu},
        "SAH": {"std": sah_std, "neu": sah_neu}
    }

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("📊 Comparação de BVH tradicional vs BVH neural\n")
        f.write("=============================================\n\n")
        f.write(f"EPO tradicional: {epo_std}\n")
        f.write(f"EPO neural:      {epo_neu}\n\n")
        f.write(f"SAH médio tradicional: {sah_std:.4f}\n")
        f.write(f"SAH médio neural:      {sah_neu:.4f}\n\n")
        f.write(json.dumps(res, indent=4))
    print(f"✅ Resultados salvos em {out_txt}")

    # ==============================================================
    # 4️⃣ Gera gráfico aprimorado (EPO + SAH normalizados)
    # ==============================================================
    labels = ["Tradicional", "Neural"]
    epo_values = [epo_std["frac"], epo_neu["frac"]]
    sah_values = [sah_std, sah_neu]

    # Normaliza SAH e EPO para [0, 1]
    max_epo = max(epo_values) if max(epo_values) > 0 else 1
    max_sah = max(sah_values) if max(sah_values) > 0 else 1
    epo_norm = [v / max_epo for v in epo_values]
    sah_norm = [v / max_sah for v in sah_values]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # Barras = EPO normalizado
    bars = ax1.bar(labels, epo_norm, color="#4B9CD3", alpha=0.7, label="EPO (fração)")
    ax1.set_ylabel("EPO normalizado", color="#4B9CD3", fontsize=10)
    ax1.tick_params(axis='y', labelcolor="#4B9CD3")

    # Linha = SAH normalizado
    ax2.plot(labels, sah_norm, color="#D34B9C", marker="o", linewidth=2.5, label="SAH (normalizado)")
    ax2.set_ylabel("SAH normalizado", color="#D34B9C", fontsize=10)
    ax2.tick_params(axis='y', labelcolor="#D34B9C")

    # Rótulos numéricos acima das barras
    for i, bar in enumerate(bars):
        val = epo_values[i]
        ax1.text(bar.get_x() + bar.get_width()/2, epo_norm[i] + 0.02, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, color="#004080")

    # Rótulos numéricos na linha SAH
    for i, val in enumerate(sah_values):
        ax2.text(i, sah_norm[i] + 0.02, f"{val:.2f}", ha="center", color="#D34B9C", fontsize=9)

    # Título e layout
    plt.title("Comparação BVH Tradicional vs Neural — EPO e SAH Normalizados", fontsize=11, weight="bold")
    fig.tight_layout()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg)
    plt.close()

    print(f"📈 Gráficos salvos em:\n   - {out_png}\n   - {out_svg}")
    return res
