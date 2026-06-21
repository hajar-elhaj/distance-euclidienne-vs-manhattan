# -*- coding: utf-8 -*-
"""Genere une image 'terminal' de la sortie du programme (pour la slide Execution)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

lines = [
    ("$ python distance_demo.py", "#6A9955"),
    ("", "#d4d4d4"),
    ("[OK] Formules validees contre SciPy", "#4EC9B0"),
    ("", "#d4d4d4"),
    ("Jeu de donnees : 20 clients, 2 variables (scores 0-10).", "#d4d4d4"),
    ("Nouveau client (a classer) : [5. 5.]", "#d4d4d4"),
    ("", "#d4d4d4"),
    ("--- Les 3 plus proches voisins ---", "#569CD6"),
    ("Euclidienne : ['C01','C02','C03'] -> ['Budget','Budget','Budget']", "#d4d4d4"),
    ("Manhattan   : ['C04','C05','C06'] -> ['Premium','Premium','Premium']", "#d4d4d4"),
    ("", "#d4d4d4"),
    ("************  RESULTAT  ************", "#DCDCAA"),
    ("  Prediction avec distance EUCLIDIENNE : Budget", "#9CDCFE"),
    ("  Prediction avec distance MANHATTAN   : Premium", "#CE9178"),
    ("  >>> MEMES donnees, MEME client, decision DIFFERENTE !", "#F48771"),
    ("***********************************", "#DCDCAA"),
]

fig, ax = plt.subplots(figsize=(11, 5.2))
fig.patch.set_facecolor("#1e1e1e")
ax.set_facecolor("#1e1e1e")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# barre de titre facon fenetre
for i, col in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.add_patch(plt.Circle((0.018 + i * 0.025, 0.95), 0.008,
                            color=col, transform=ax.transAxes, clip_on=False))
ax.text(0.5, 0.95, "Terminal  —  distance_demo.py", color="#9d9d9d",
        fontsize=11, ha="center", va="center", family="monospace")

y = 0.86
for text, color in lines:
    ax.text(0.02, y, text, color=color, fontsize=12.5,
            family="monospace", va="top")
    y -= 0.052

fig.savefig("figures/fig3_execution.png", dpi=150,
            facecolor="#1e1e1e", bbox_inches="tight", pad_inches=0.25)
print("Image creee : figures/fig3_execution.png")
