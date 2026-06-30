"""
Football Oracle - Calibration LIVE (corrige les biais mesurés sur la Coupe du Monde)
====================================================================================

Le radar montre des biais SYSTEMATIQUES du modele sur les matchs WC deja joues :
  - nuls SOUS-estimes (ex. P(nul) 22% vs 29% reel, et 18% vs 38% sur gros favoris)
  - favoris parfois SUR-estimes
On corrige les probabilites du modele en les NUDGEANT vers ce que la realite live
montre, de facon AMORTIE (sur ~48 matchs c'est bruite -> on n'applique qu'une
fraction alpha du biais) et BORNEE. Auto-derive de data/mpp_live_tracking.csv :
ca se met a jour tout seul a chaque journee, et ca s'annule si le biais disparait.

Honnetete : derive du PASSE (matchs joues), valide sur le FUTUR (le radar dira si
le biais persiste). Ne pas confondre avec une amelioration in-sample.

Pur stdlib (csv, os). ISOLATION : aucun fichier NBA.
"""
import csv
import os

# classe MPP : 'T1' <-> ACTUAL_CLASS 2, 'NUL' <-> 1, 'T2' <-> 0
_ACT = {'T1': 2, 'NUL': 1, 'T2': 0}
_PCOL = {'T1': 'PROB_T1', 'NUL': 'PROB_NUL', 'T2': 'PROB_T2'}


def load_calibration(csv_path='data/mpp_live_tracking.csv', min_n=30, alpha=0.5,
                     max_shift=0.10, half_life=12):
    """
    Renvoie {'bias':{T1,NUL,T2}, 'n', 'eff_n', 'alpha'} ou None si pas assez de matchs.
    bias_k = alpha * (freq_reelle_k - proba_moyenne_predite_k), borne a +/- max_shift.

    PONDERE PAR RECENCE : les matchs RECENTS pesent plus (poids divise par 2 tous les
    `half_life` matchs). -> s'adapte vite au changement de regime 90' (poules) -> 120'
    (phase finale, ou les nuls sont PLUS RARES car la prolongation casse les egalites).
    """
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    'date': str(r.get('DATE', '')),
                    'act': int(float(r['ACTUAL_CLASS'])),
                    'p': {k: float(r[_PCOL[k]]) for k in _ACT},
                })
            except (KeyError, ValueError):
                continue
    n = len(rows)
    if n < min_n:
        return None
    rows.sort(key=lambda r: r['date'])                  # chronologique : récents en dernier
    decay = 0.5 ** (1.0 / max(1, half_life))
    w = [decay ** (n - 1 - i) for i in range(n)]        # récent -> ~1, ancien -> décroît
    W = sum(w)
    bias = {}
    for k, code in _ACT.items():
        mean_pred = sum(w[i] * rows[i]['p'][k] for i in range(n)) / W
        freq = sum(w[i] * (1.0 if rows[i]['act'] == code else 0.0) for i in range(n)) / W
        raw = alpha * (freq - mean_pred)
        bias[k] = max(-max_shift, min(max_shift, raw))
    eff_n = (W * W) / sum(x * x for x in w)             # taille d'échantillon effective (Kish)
    return {'bias': bias, 'n': n, 'eff_n': eff_n, 'alpha': alpha, 'half_life': half_life}


def correct(p_model, calib):
    """Applique le nudge amorti puis renormalise. p_model et retour : dict {T1,NUL,T2}."""
    if not calib:
        return dict(p_model)
    p = {k: min(0.98, max(0.01, p_model[k] + calib['bias'][k])) for k in p_model}
    s = sum(p.values())
    return {k: p[k] / s for k in p}


def describe(calib):
    if not calib:
        return "calibration live : ❌ inactive (< matchs requis)"
    b = calib['bias']
    eff = calib.get('eff_n', calib['n'])
    return (f"calibration live : ✅ active (n={calib['n']}, ~{eff:.0f} récents pondérés, α={calib['alpha']}) | "
            f"nudge T1 {b['T1']*100:+.0f} / NUL {b['NUL']*100:+.0f} / T2 {b['T2']*100:+.0f}")


# ---------------------------------- auto-test ----------------------------------
if __name__ == "__main__":
    c = load_calibration()
    print(describe(c))
    if c:
        for raw in [{'T1': .72, 'NUL': .18, 'T2': .10}, {'T1': .34, 'NUL': .25, 'T2': .41}]:
            adj = correct(raw, c)
            print(f"  brut {{'T1':{raw['T1']:.2f},'NUL':{raw['NUL']:.2f},'T2':{raw['T2']:.2f}}} "
                  f"-> calibré {{'T1':{adj['T1']:.2f},'NUL':{adj['NUL']:.2f},'T2':{adj['T2']:.2f}}}")
