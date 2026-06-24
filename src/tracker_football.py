"""
Football Oracle - Radar de performance LIVE (Coupe du Monde 2026)
=================================================================

Deux parties :
  1) run_world_cup_tracking() : rejoue le modèle CALIBRÉ sur les matchs WC déjà
     joués -> data/mpp_live_tracking.csv (nécessite pandas/joblib/xgboost).
  2) print_radar() : rapport STDLIB lu depuis ce CSV (log loss live vs baseline,
     Brier, calibration par classe, check "gros favoris") -> exécutable sans modèle.

Usage :
  python src/tracker_football.py            # régénère le CSV puis imprime le radar
  python src/tracker_football.py --radar    # imprime le radar depuis le CSV existant

ISOLATION : aucun fichier NBA.
"""
import os
import sys
import csv
import math
from collections import Counter

CSV_PATH = 'data/mpp_live_tracking.csv'
CLASSES = {0: 'T2 (ext)', 1: 'NUL', 2: 'T1 (dom)'}


# ============================== GÉNÉRATION (modèle) ==============================
def run_world_cup_tracking():
    import pandas as pd
    import numpy as np
    import joblib
    from mpp_oracle import prepare_live_inference_features

    print("=" * 60)
    print("📡 RADAR DE PERFORMANCE (COUPE DU MONDE 2026)")
    print("=" * 60)

    results_path = 'data/wc_2026_results.csv'
    if not os.path.exists(results_path):
        print("❌ Fichier de résultats de la Coupe du Monde introuvable.")
        return

    wc_df = pd.read_csv(results_path, names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
    wc_df = wc_df.dropna(subset=['HG', 'AG'])                      # seulement les matchs joués
    history_df = pd.read_csv('data/processed_football_games.csv')

    # Modèle CALIBRÉ (cohérent avec la page) ; repli sur le brut si absent
    model_path = 'models/football_classifier_calibrated.pkl'
    if not os.path.exists(model_path):
        model_path = 'models/football_classifier.pkl'
        print("⚠️ modèle calibré absent -> radar sur le modèle brut (relance train_football_models.py)")
    clf = joblib.load(model_path)
    with open('models/football_features.txt', 'r') as f:
        features_list = [line.strip() for line in f.read().split(',') if line.strip()]

    predictions_log, missing_teams = [], set()
    print(f"🔍 Analyse de {len(wc_df)} matchs réels joués...")

    for _, row in wc_df.iterrows():
        match_date = str(row['DATE'])
        t1, t2 = row['HOME'], row['AWAY']
        hg, ag = row['HG'], row['AG']
        target = 2 if hg > ag else (1 if hg == ag else 0)

        elo_dict = {}
        for t in [t1, t2]:
            team_hist = history_df[(history_df['TEAM_ID'] == t) &
                                   (history_df['MATCH_DATE'] < match_date)].sort_values('MATCH_DATE')
            if team_hist.empty:
                missing_teams.add(t)                              # guard noms (E3)
                elo_dict[t] = 1500
            else:
                elo_dict[t] = team_hist.iloc[-1]['ELO_POST']

        X_pred = prepare_live_inference_features(t1, t2, match_date, history_df, elo_dict)
        X_pred['T1_IS_HOST'] = 0
        X_pred = X_pred[features_list]
        probs = clf.predict_proba(X_pred)[0]                      # [T2, NUL, T1]

        predictions_log.append({
            'DATE': match_date, 'MATCH': f"{t1} vs {t2}",
            'PROB_T1': probs[2], 'PROB_NUL': probs[1], 'PROB_T2': probs[0],
            'PREDICTED_CLASS': int(np.argmax(probs)), 'ACTUAL_CLASS': target,
        })

    pd.DataFrame(predictions_log).to_csv(CSV_PATH, index=False)
    if missing_teams:
        print(f"⚠️ Équipes absentes de l'historique (ELO=1500 par défaut, vérifier les noms) : "
              f"{sorted(missing_teams)}")
    print(f"💾 {CSV_PATH} régénéré ({len(predictions_log)} matchs).")
    print_radar(CSV_PATH)


# ============================== RADAR (stdlib pur) ==============================
def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    recs = []
    for r in rows:
        recs.append({
            'p': {2: float(r['PROB_T1']), 1: float(r['PROB_NUL']), 0: float(r['PROB_T2'])},
            'pred': int(float(r['PREDICTED_CLASS'])),
            'act': int(float(r['ACTUAL_CLASS'])),
            'match': r.get('MATCH', '?'),
        })
    return recs


def print_radar(path=CSV_PATH):
    if not os.path.exists(path):
        print(f"❌ {path} introuvable (lance d'abord la génération).")
        return
    recs = _read_csv(path)
    n = len(recs)
    if n == 0:
        print("Radar : aucune donnée.")
        return
    eps = 1e-15

    acc = sum(1 for r in recs if r['pred'] == r['act']) / n
    ll = -sum(math.log(max(r['p'][r['act']], eps)) for r in recs) / n
    act_c = Counter(r['act'] for r in recs)
    base = {k: act_c.get(k, 0) / n for k in (0, 1, 2)}
    llb = -sum(math.log(max(base[r['act']], eps)) for r in recs) / n
    brier = sum(sum((r['p'][k] - (1.0 if r['act'] == k else 0.0)) ** 2 for k in (0, 1, 2))
                for r in recs) / n / 3.0

    print("\n" + "=" * 60)
    print(f"📈 RADAR LIVE — {n} matchs WC")
    print("=" * 60)
    print(f"  Accuracy 1X2 : {acc:.1%}")
    edge = llb - ll
    print(f"  Log loss     : {ll:.4f}  vs baseline {llb:.4f}  -> edge {edge:+.4f} "
          f"({'positif' if edge > 0 else 'NÉGATIF'})")
    print(f"  Brier (moy.) : {brier:.4f}")

    print("\n  Par classe (recall + calibration proba moyenne vs fréquence réelle) :")
    for k in (2, 1, 0):
        nb = act_c.get(k, 0)
        rec_k = [r for r in recs if r['act'] == k]
        recall = (sum(1 for r in rec_k if r['pred'] == k) / nb) if nb else 0.0
        mean_p = sum(r['p'][k] for r in recs) / n          # proba moyenne assignée à la classe k
        freq = base[k]
        flag = ""
        if k == 1 and abs(mean_p - freq) > 0.04:
            flag = ("  ⚠️ P(nul) surévaluée" if mean_p > freq
                    else "  ⚠️ P(nul) SOUS-évaluée → les nuls sont un filon (lean draws)")
        print(f"    {CLASSES[k]:<9} : n={nb:<3} recall={recall:4.0%}  "
              f"P_moy={mean_p:4.0%} vs réel={freq:4.0%}{flag}")

    # Check "gros favoris" : le modèle voit un favori net (max(T1,T2) >= 0.60)
    fav = [r for r in recs if max(r['p'][2], r['p'][0]) >= 0.60]
    if fav:
        side = lambda r: 2 if r['p'][2] >= r['p'][0] else 0
        win_rate = sum(1 for r in fav if r['act'] == side(r)) / len(fav)
        mean_fav_p = sum(max(r['p'][2], r['p'][0]) for r in fav) / len(fav)
        mean_nul_p = sum(r['p'][1] for r in fav) / len(fav)
        real_nul = sum(1 for r in fav if r['act'] == 1) / len(fav)
        fav_flag = "  ⚠️ modèle trop confiant sur les favoris" if mean_fav_p - win_rate > 0.05 else ""
        print(f"\n  Gros favoris (P≥60%, n={len(fav)}) : le favori gagne {win_rate:.0%} "
              f"(prédit {mean_fav_p:.0%}){fav_flag}")
        if mean_nul_p - real_nul > 0.05:
            nul_flag = "  ⚠️ nul surévalué → prudence sur 'nul contrarian' contre un gros favori"
        elif real_nul - mean_nul_p > 0.05:
            nul_flag = "  ⚠️ nul SOUS-évalué → sur gros favori le nul tombe + souvent que prévu (bon spot contrarian)"
        else:
            nul_flag = "  ✅ cohérent"
        print(f"    P(nul) moyenne prédite {mean_nul_p:.0%} vs nuls réels {real_nul:.0%}{nul_flag}")

    # Verdict synthétique
    print("\n  VERDICT :")
    print(f"    • Edge live {'réel mais à surveiller' if 0 < edge < 0.05 else ('positif' if edge >= 0.05 else 'NÉGATIF (probas à recalibrer)')}.")
    print("    • Pilote par E[points]/edge vs foule (pas l'argmax) ; relance ce radar après chaque journée.")
    print("=" * 60)


if __name__ == "__main__":
    if '--radar' in sys.argv:
        print_radar()
    else:
        run_world_cup_tracking()
