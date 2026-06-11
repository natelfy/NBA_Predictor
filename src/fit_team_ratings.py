"""
Football Oracle - Forces d'equipe ajustees a l'adversaire (Poisson regression)
==============================================================================

CORRIGE le biais des EMA : une equipe qui a joue des adversaires faibles
paraissait trop forte. Ici, regression de Poisson :

    log E[buts] = intercept + attaque[equipe] + defense[adversaire] + host*IS_HOST

Chaque equipe obtient une note d'attaque ET une note de defense, estimees
EN TENANT COMPTE du niveau des adversaires rencontres. Sauve data/team_ratings.json.

ISOLATION : aucun fichier NBA. Usage : python src/fit_team_ratings.py
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

DATA = 'data/processed_football_games.csv'
OUT = 'data/team_ratings.json'

def fit(years=6, alpha=0.005, verbose=True):
    df = pd.read_csv(DATA)
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    cutoff = df['MATCH_DATE'].max() - pd.DateOffset(years=years)
    df = df[df['MATCH_DATE'] >= cutoff].dropna(
        subset=['GOALS', 'TEAM_ID', 'OPP_TEAM_ID', 'IS_HOST']).copy()
    df['GOALS'] = df['GOALS'].astype(float)

    enc_a = OneHotEncoder(handle_unknown='ignore')
    enc_d = OneHotEncoder(handle_unknown='ignore')
    A = enc_a.fit_transform(df[['TEAM_ID']])
    D = enc_d.fit_transform(df[['OPP_TEAM_ID']])
    H = csr_matrix(df[['IS_HOST']].astype(float).values)
    X = hstack([A, D, H]).tocsr()
    y = df['GOALS'].values

    m = PoissonRegressor(alpha=alpha, max_iter=1000)
    m.fit(X, y)

    atk_teams = list(enc_a.categories_[0])
    def_teams = list(enc_d.categories_[0])
    na = len(atk_teams)
    coef = m.coef_
    attack = {t: float(coef[i]) for i, t in enumerate(atk_teams)}
    defense = {t: float(coef[na + i]) for i, t in enumerate(def_teams)}
    out = {'intercept': float(m.intercept_), 'host_coef': float(coef[na + len(def_teams)]),
           'attack': attack, 'defense': defense}
    os.makedirs('data', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f)

    if verbose:
        print(f"Ajuste sur {len(df)} lignes-equipe ({years} ans). Sauve : {OUT}")
        top_atk = sorted(attack.items(), key=lambda x: -x[1])[:5]
        top_def = sorted(defense.items(), key=lambda x: x[1])[:5]  # def basse = solide
        print("  Meilleures attaques :", [t for t, _ in top_atk])
        print("  Meilleures defenses :", [t for t, _ in top_def])
    return out

if __name__ == "__main__":
    fit()

