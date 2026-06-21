# -*- coding: utf-8 -*-
"""
====================================================================
 Sujet 13 : Distance Euclidienne vs Manhattan
 Python Avance - Master Informatique et Telecommunications (S2)
--------------------------------------------------------------------
 Demonstration AVANT / APRES exigee par le guide :
   AVANT : distance arbitraire -> on ne sait pas qui est "voisin".
   APRES : l'application de la metrique change le VOISINAGE,
           et donc la prediction du classifieur KNN.
====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # backend fichier (pas besoin d'ecran)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

# Validation independante (on prouve que nos formules sont correctes)
from scipy.spatial import distance as scipy_distance

DATA_PATH = "data/clients.csv"
FIG_DIR = "figures"
K = 3                            # nombre de voisins pour le KNN
QUERY = np.array([5.0, 5.0])     # nouveau client a classer (point inconnu)


# ====================================================================
# 1. LES DEUX FORMULES, IMPLEMENTEES A LA MAIN
# ====================================================================
def distance_euclidienne(p, q):
    """Norme L2 : racine de la somme des carres des differences.
       d = sqrt( somme_i (p_i - q_i)^2 )"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.sqrt(np.sum((p - q) ** 2))


def distance_manhattan(p, q):
    """Norme L1 : somme des valeurs absolues des differences.
       d = somme_i |p_i - q_i|"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.sum(np.abs(p - q))


def verifier_formules():
    """On compare nos fonctions a SciPy pour prouver leur exactitude."""
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert np.isclose(distance_euclidienne(a, b), scipy_distance.euclidean(a, b))
    assert np.isclose(distance_manhattan(a, b), scipy_distance.cityblock(a, b))
    print("[OK] Formules validees contre SciPy  "
          "(Euclide(0,0->3,4)=5.0 ; Manhattan=7.0)\n")


# ====================================================================
# 2. KNN "maison" : on classe un point selon ses K plus proches voisins
# ====================================================================
def knn_predire(query, X, y, k, fonction_distance):
    """Retourne (prediction, tableau_des_distances_trie)."""
    distances = np.array([fonction_distance(query, point) for point in X])
    ordre = np.argsort(distances)                 # indices du plus proche au plus loin
    voisins = ordre[:k]
    votes = pd.Series(y[voisins]).value_counts()  # vote majoritaire
    prediction = votes.index[0]
    return prediction, distances, voisins


# ====================================================================
# 3. PROGRAMME PRINCIPAL
# ====================================================================
def main():
    print("=" * 64)
    print(" SUJET 13 : DISTANCE EUCLIDIENNE vs MANHATTAN")
    print("=" * 64, "\n")

    verifier_formules()

    # --- Chargement du jeu de donnees ---
    df = pd.read_csv(DATA_PATH)
    X = df[["spending_score", "frequency_score"]].to_numpy()
    y = df["segment"].to_numpy()
    print("Jeu de donnees : %d clients, 2 variables (scores 0-10)." % len(df))
    print("Probleme metier : a quel segment appartient un NOUVEAU client ?")
    print("Nouveau client (a classer) :", QUERY, "\n")

    # ----------------------------------------------------------------
    # AVANT : aucune metrique -> la notion de "voisin" est arbitraire
    # ----------------------------------------------------------------
    print("-" * 64)
    print(" AVANT  ->  sans metrique, impossible de decider du voisinage")
    print("-" * 64)
    print("On voit un nuage de points et un nouveau client, mais 'proche'")
    print("n'a aucun sens tant qu'on n'a pas choisi une distance.\n")

    # ----------------------------------------------------------------
    # APRES : on applique les deux metriques sur LES MEMES donnees
    # ----------------------------------------------------------------
    print("-" * 64)
    print(" APRES  ->  la metrique definit le voisinage ET la decision")
    print("-" * 64)

    pred_e, dist_e, voisins_e = knn_predire(QUERY, X, y, K, distance_euclidienne)
    pred_m, dist_m, voisins_m = knn_predire(QUERY, X, y, K, distance_manhattan)

    # Tableau comparatif des distances (les memes points, deux mesures)
    comp = pd.DataFrame({
        "id": df["id"],
        "segment": df["segment"],
        "d_Euclidienne": np.round(dist_e, 3),
        "d_Manhattan": np.round(dist_m, 3),
    }).sort_values("d_Euclidienne").reset_index(drop=True)
    print("\nDistances du nouveau client a chaque client (triees par Euclide) :")
    print(comp.head(8).to_string(index=False))

    print("\n--- Les %d plus proches voisins ---" % K)
    print("Euclidienne :", list(df['id'].to_numpy()[voisins_e]),
          "->", list(y[voisins_e]))
    print("Manhattan   :", list(df['id'].to_numpy()[voisins_m]),
          "->", list(y[voisins_m]))

    print("\n" + "*" * 64)
    print(" RESULTAT (impact geometrique sur le voisinage)")
    print("*" * 64)
    print("  Prediction avec distance EUCLIDIENNE : %s" % pred_e)
    print("  Prediction avec distance MANHATTAN   : %s" % pred_m)
    if pred_e != pred_m:
        print("  >>> MEMES donnees, MEME client, mais decision DIFFERENTE !")
        print("  >>> Le choix de la metrique change le resultat du modele.")
    print("*" * 64 + "\n")

    # --- Generation des figures pour les diapositives ---
    figure_balls_unitaires()
    figure_avant(df)
    figure_apres(df, voisins_e, voisins_m, dist_e, dist_m, pred_e, pred_m)
    print("3 figures enregistrees dans le dossier 'figures/'.\n")


# ====================================================================
# 4. FIGURES
# ====================================================================
def _style_axes(ax):
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("Spending score")
    ax.set_ylabel("Frequency score")


def figure_balls_unitaires():
    """Figure pedagogique : 'boule' de rayon 1 pour chaque metrique.
       Euclide = cercle ; Manhattan = losange."""
    fig, ax = plt.subplots(figsize=(6, 6))
    centre = (0, 0)
    ax.add_patch(Circle(centre, 1.0, fill=False, lw=2.5,
                        edgecolor="#1f77b4", label="Euclidienne (cercle)"))
    # losange = carre tourne de 45 deg, "rayon" 1
    ax.add_patch(RegularPolygon(centre, numVertices=4, radius=1.0,
                                orientation=0, fill=False, lw=2.5,
                                edgecolor="#d62728", label="Manhattan (losange)"))
    ax.plot(0, 0, "k.", ms=8)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_title("Tous les points a distance 1 du centre")
    _style_axes(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig0_boules_unitaires.png", dpi=150)
    plt.close(fig)


def _scatter_clients(ax, df):
    for seg, color, marker in [("Budget", "#2ca02c", "o"),
                               ("Premium", "#9467bd", "s")]:
        sub = df[df["segment"] == seg]
        ax.scatter(sub["spending_score"], sub["frequency_score"],
                   c=color, marker=marker, s=70, edgecolors="k",
                   linewidths=0.5, label=seg, zorder=3)
    ax.scatter(*QUERY, c="red", marker="*", s=420, edgecolors="k",
               linewidths=1, label="Nouveau client", zorder=5)


def figure_avant(df):
    """AVANT : donnees brutes, aucune notion de voisinage."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _scatter_clients(ax, df)
    ax.set_title("AVANT : qui sont les voisins du nouveau client ?\n"
                 "(sans metrique, la reponse est arbitraire)")
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5)
    _style_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_avant.png", dpi=150)
    plt.close(fig)


def figure_apres(df, voisins_e, voisins_m, dist_e, dist_m, pred_e, pred_m):
    """APRES : deux panneaux montrant que le voisinage (et la decision) change."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    X = df[["spending_score", "frequency_score"]].to_numpy()

    # --- Panneau Euclidien : cercle passant par le K-ieme voisin ---
    ax = axes[0]
    _scatter_clients(ax, df)
    rayon_e = np.sort(dist_e)[K - 1]
    ax.add_patch(Circle(QUERY, rayon_e, fill=False, lw=2,
                        edgecolor="#1f77b4", ls="--"))
    ax.scatter(X[voisins_e, 0], X[voisins_e, 1], s=260, facecolors="none",
               edgecolors="#1f77b4", linewidths=2.5, zorder=6)
    ax.set_title("APRES - Distance EUCLIDIENNE\nPrediction : %s" % pred_e)
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5)
    _style_axes(ax); ax.legend(loc="upper left")

    # --- Panneau Manhattan : losange passant par le K-ieme voisin ---
    ax = axes[1]
    _scatter_clients(ax, df)
    rayon_m = np.sort(dist_m)[K - 1]
    ax.add_patch(RegularPolygon(QUERY, numVertices=4, radius=rayon_m,
                                orientation=0, fill=False, lw=2,
                                edgecolor="#d62728", ls="--"))
    ax.scatter(X[voisins_m, 0], X[voisins_m, 1], s=260, facecolors="none",
               edgecolors="#d62728", linewidths=2.5, zorder=6)
    ax.set_title("APRES - Distance MANHATTAN\nPrediction : %s" % pred_m)
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5)
    _style_axes(ax); ax.legend(loc="upper left")

    fig.suptitle("Memes donnees, metrique differente -> voisinage et decision differents",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_apres.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
