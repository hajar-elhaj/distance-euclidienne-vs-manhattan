# Sujet 13 — Distance Euclidienne vs Manhattan

Présentation Python Avancé (Master I&T, S2).

## Lancer la démo
```bash
pip install -r requirements.txt
python distance_demo.py
```

## Contenu
- `data/clients.csv` — jeu de données : 20 clients, 2 variables (scores 0–10), 2 segments.
- `distance_demo.py` — formules implémentées à la main + validation SciPy + KNN + figures.
- `figures/` — images générées pour les diapositives :
  - `fig0_boules_unitaires.png` — cercle (Euclide) vs losange (Manhattan).
  - `fig1_avant.png` — AVANT : données brutes, voisinage indéfini.
  - `fig2_apres.png` — APRÈS : le voisinage et la prédiction changent selon la métrique.

## Le résultat clé
Pour le même nouveau client (5, 5) et K=3 :
- Distance **Euclidienne** → 3 voisins *Budget* → prédiction **Budget**
- Distance **Manhattan** → 3 voisins *Premium* → prédiction **Premium**

Mêmes données, métrique différente → décision différente.
