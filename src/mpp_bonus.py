"""
Football Oracle - Bonus tournoi v3 (picks a figer AVANT le coup d'envoi)
========================================================================

CHANGEMENTS vs v2 (qui avait produit "Argentine = pire equipe", absurde) :
  - PLUS d'inner join : le scrape d'effectifs n'a ramene que 5 equipes, donc
    v2 reduisait l'univers aux 5 meilleures nations -> "pire" = Argentine.
    Ici on classe TOUTES les equipes qualifiees presentes dans le modele.
  - La valeur d'effectif (scrape casse) est ABANDONNEE comme signal. On la
    remplace par la COTE DU MARCHE, bien meilleur predicteur et deja agrege.
  - Garde-fou : si < 30 equipes exploitables, on s'arrete (donnees incompletes).
  - "Surprise" et "pire equipe" calculees sur le pool COMPLET, pas un sous-pool.

SIGNAUX :
  1. Force modele (Base_Prob) : proba de battre une elite moyenne, terrain neutre.
  2. Marche : proba implicite des bookmakers (snapshot a mettre a jour).

ISOLATION : aucun fichier NBA touche. Lecture seule du modele foot.
Usage : python src/mpp_bonus_v3.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

QUALIFIED_TEAMS = [
    'Canada', 'United States', 'Mexico', 'Saudi Arabia', 'Australia', 'Iraq', 'Japan', 'Jordan',
    'Uzbekistan', 'Qatar', 'South Korea', 'Iran', 'South Africa', 'Algeria', 'Cape Verde',
    'Ivory Coast', "Côte d'Ivoire", 'Egypt', 'Ghana', 'Morocco', 'DR Congo', 'Senegal', 'Tunisia',
    'Curaçao', 'Haiti', 'Panama', 'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
    'New Zealand', 'Germany', 'England', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia',
    'Scotland', 'France', 'Spain', 'Norway', 'Netherlands', 'Portugal', 'Sweden', 'Switzerland',
    'Czechia', 'Czech Republic', 'Turkey',
]

# [A VERIFIER] SNAPSHOT marche ~10 juin 2026 (FanDuel/BetMGM/DraftKings, moyenne).
# Proba implicite BRUTE (vig inclus) : sert au classement relatif, PAS comme
# vraie proba. A REACTUALISER juste avant de soumettre (les cotes bougent).
MARKET_PROB = {
    'Spain': 0.179, 'France': 0.169, 'England': 0.133, 'Brazil': 0.108,
    'Portugal': 0.103, 'Argentina': 0.095, 'Germany': 0.067, 'Netherlands': 0.048,
    'Morocco': 0.020, 'Mexico': 0.018, 'United States': 0.016, 'Canada': 0.004,
}


def build_model_power():
    clf = joblib.load('models/football_classifier.pkl')
    with open('models/football_features.txt') as f:
        features = f.read().split(',')
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    recent = df[df['MATCH_DATE'] >= df['MATCH_DATE'].max() - pd.Timedelta(days=730)]
    latest = recent.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()
    latest = latest[latest['TEAM_ID'].isin(QUALIFIED_TEAMS)]

    top5 = latest.nlargest(5, 'ELO_POST')
    base_feats = [f.replace('_DIFF', '') for f in features if f != 'T1_IS_HOST']
    boss = top5[base_feats].mean()

    rows = []
    for _, t in latest.iterrows():
        pred = {}
        for feat in features:
            if feat == 'T1_IS_HOST':
                pred[feat] = [0]
            else:
                bf = feat.replace('_DIFF', '')
                pred[feat] = [t[bf] - boss[bf]] if bf in t else [0]
        X = pd.DataFrame(pred)[features]
        rows.append({'Team': t['TEAM_ID'], 'ELO': t['ELO_POST'],
                     'Form_10': t.get('FORM_10', np.nan),
                     'Base_Prob': clf.predict_proba(X)[0][2]})
    return pd.DataFrame(rows)


def main():
    print("=" * 66)
    print("FOOTBALL ORACLE - BONUS v3 (classement complet, sans scrape)")
    print("=" * 66)

    df = build_model_power()
    n = len(df)
    print(f"\nEquipes classees : {n}")
    # GARDE-FOU : on ne produit JAMAIS de pick sur un pool tronque.
    if n < 30:
        print(f"[STOP] Seulement {n} equipes dans le modele (attendu ~48).")
        print("   Donnees incompletes : verifie QUALIFIED_TEAMS et processed_football_games.csv.")
        print("   Aucun pick produit (mieux vaut rien que de l'absurde).")
        return

    df['model_rank'] = df['Base_Prob'].rank(ascending=False, method='min').astype(int)
    df['mkt'] = df['Team'].map(MARKET_PROB)
    df['market_rank'] = df['mkt'].rank(ascending=False, method='min')
    df = df.sort_values('model_rank')

    # --- Classement complet (transparence : tu sanity-check tout) ---
    print("\n" + "=" * 66)
    print("CLASSEMENT MODELE COMPLET (force = proba battre une elite, neutre)")
    print("=" * 66)
    print(f"\n   {'#':>2} {'Equipe':<22} {'ELO':>5} {'Force':>6} {'Marche':>7} {'rMkt':>5}")
    print("   " + "-" * 52)
    for _, r in df.iterrows():
        mkt = f"{r['mkt']:.1%}" if pd.notna(r['mkt']) else "  -"
        rmkt = f"{int(r['market_rank'])}" if pd.notna(r['market_rank']) else "-"
        print(f"   {r['model_rank']:>2} {r['Team']:<22} {r['ELO']:>5.0f} "
              f"{r['Base_Prob']:>6.1%} {mkt:>7} {rmkt:>5}")

    # =================== PICKS ===================
    print("\n" + "=" * 66)
    print("PICKS A SOUMETTRE")
    print("=" * 66)

    # [1] VAINQUEUR : top modele, cross-check marche
    win = df.iloc[0]
    mkt_fav = df.loc[df['mkt'].idxmax()] if df['mkt'].notna().any() else None
    print(f"\n[1] VAINQUEUR : {win['Team'].upper()}  (force {win['Base_Prob']:.1%}, ELO {win['ELO']:.0f})")
    if mkt_fav is not None:
        if mkt_fav['Team'] == win['Team']:
            print(f"    >> Modele ET marche d'accord : signal le plus solide possible.")
        else:
            print(f"    [A VERIFIER] Le marche prefere {mkt_fav['Team']} ({mkt_fav['mkt']:.1%}). "
                  f"Divergence -> arbitre selon les points MPP offerts sur chaque option.")
    print("    [A VERIFIER] Indice de force, pas une proba de titre (ignore le tirage).")
    print("    Pour durcir : simulation Monte-Carlo du tableau (groupes -> KO).")

    # [2] SURPRISE : meilleure equipe HORS top-8, value vs marche si dispo
    favorites = set(df.head(8)['Team'])
    outs = df[~df['Team'].isin(favorites)].copy()
    # priorite : forte dynamique recente (Form_10) parmi les outsiders credibles
    outs = outs[outs['Base_Prob'] >= outs['Base_Prob'].median()]
    surprise = outs.sort_values('Form_10', ascending=False).iloc[0] if len(outs) else df.iloc[8]
    print(f"\n[2] EQUIPE SURPRISE : {surprise['Team'].upper()}  "
          f"(rang modele {int(surprise['model_rank'])}, ELO {surprise['ELO']:.0f}, "
          f"Form_10 {surprise['Form_10']:.2f})")
    print("    Cross-check marche : dark horses cotes par les books = Netherlands (20/1),")
    print("    Morocco (50/1, 1/2 finaliste 2022). Compare avec le pick ci-dessus.")

    # [3] PIRE EQUIPE : plus faible force sur le POOL COMPLET
    worst = df.iloc[-1]
    print(f"\n[3] PIRE EQUIPE : {worst['Team'].upper()}  "
          f"(force {worst['Base_Prob']:.1%}, ELO {worst['ELO']:.0f}, derniere sur {n})")

    # [4] MEILLEUR BUTEUR : non modelisable honnetement
    top3 = df.head(3)['Team'].tolist()
    print(f"\n[4] MEILLEUR BUTEUR :")
    print(f"    [A VERIFIER] Aucun modele joueur -> non predictible avec ces donnees.")
    print(f"    Heuristique : buteur focal d'une equipe du top 3 ({', '.join(top3)}).")
    print(f"    Cross-check marche utile : regarde les cotes 'Golden Boot' des books.")
    print("=" * 66)


if __name__ == "__main__":
    main()