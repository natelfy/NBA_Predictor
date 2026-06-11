"""
Football Oracle - Moteur de score exact + EV par match (MPP)
============================================================

C'EST LE COEUR DU SYSTEME. Les 104 matchs valent ~95% des points de ta ligue.

PRINCIPE MPP : tu joues un score exact. Tu gagnes la cote du RESULTAT (1/N/2)
si le resultat est bon, + un bonus de rarete si le score exact est bon.
  EV(jouer un score) = cote_resultat x P(resultat) + bonus x P(score exact)

FORCES D'EQUIPE : si data/team_ratings.json existe (produit par
fit_team_ratings.py, ajuste a l'adversaire), on l'utilise EN PRIORITE.
Sinon on retombe sur les EMA (non ajustees -> moins fiable).

ISOLATION : aucun fichier NBA. Usage :
  python src/fit_team_ratings.py            # 1x : calcule les forces ajustees
  python src/mpp_match_engine.py "Mexico" "South Africa" --cote-a 49 --cote-draw 125 --cote-b 148
  python src/mpp_match_engine.py --demo
  modes : equilibre (defaut) | leader (tu menes) | challenger (tu chasses)
"""
import os, math, json, argparse

# ------------------------- POISSON -------------------------

def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)

def score_matrix(lam_a, lam_b, max_goals=8):
    pa = [pois(i, lam_a) for i in range(max_goals + 1)]
    pb = [pois(j, lam_b) for j in range(max_goals + 1)]
    return [[pa[i] * pb[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]

def result_probs(mat):
    n = len(mat)
    p_a = sum(mat[i][j] for i in range(n) for j in range(n) if i > j)
    p_d = sum(mat[i][i] for i in range(n))
    p_b = sum(mat[i][j] for i in range(n) for j in range(n) if i < j)
    return p_a, p_d, p_b

# ------------------------- FORCES -------------------------

def load_ratings():
    if os.path.exists('data/team_ratings.json'):
        with open('data/team_ratings.json') as f:
            return ('reg', json.load(f))
    import pandas as pd
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    recent = df[df['MATCH_DATE'] >= df['MATCH_DATE'].max() - pd.Timedelta(days=730)]
    latest = recent.sort_values('MATCH_DATE').groupby('TEAM_ID').last()
    latest = latest.dropna(subset=['AVG_GOALS_10', 'AVG_OPP_GOALS_10'])
    league_avg = latest['AVG_GOALS_10'].mean()
    ratings = {t: (r['AVG_GOALS_10'] / league_avg, r['AVG_OPP_GOALS_10'] / league_avg)
               for t, r in latest.iterrows()}
    return ('ema', ratings, league_avg)

def teams_available(robj):
    if robj[0] == 'reg':
        return set(robj[1]['attack']) & set(robj[1]['defense'])
    return set(robj[1])

def build_lambdas(team_a, team_b, robj, host=None):
    if robj[0] == 'reg':
        R = robj[1]
        ic, hc = R['intercept'], R['host_coef']
        ha = hc if host == 'a' else 0.0
        hb = hc if host == 'b' else 0.0
        lam_a = math.exp(ic + R['attack'][team_a] + R['defense'][team_b] + ha)
        lam_b = math.exp(ic + R['attack'][team_b] + R['defense'][team_a] + hb)
        return lam_a, lam_b
    _, ratings, league_avg = robj
    atk_a, dfn_a = ratings[team_a]
    atk_b, dfn_b = ratings[team_b]
    lam_a = league_avg * atk_a * dfn_b * (1.10 if host == 'a' else 1.0)
    lam_b = league_avg * atk_b * dfn_a * (1.10 if host == 'b' else 1.0)
    return lam_a, lam_b

# ------------------------- EV + RECO -------------------------

def recommend(team_a, team_b, lam_a, lam_b, cotes, mode='equilibre', max_goals=8):
    mat = score_matrix(lam_a, lam_b, max_goals)
    p_a, p_d, p_b = result_probs(mat)
    ev = {'a': cotes['a'] * p_a, 'draw': cotes['draw'] * p_d, 'b': cotes['b'] * p_b}
    prob = {'a': p_a, 'draw': p_d, 'b': p_b}
    label = {'a': team_a, 'draw': 'Nul', 'b': team_b}

    print("\n" + "=" * 60)
    print(f"  {team_a}  vs  {team_b}")
    print("=" * 60)
    print(f"  Buts attendus : {team_a} {lam_a:.2f} | {team_b} {lam_b:.2f}")
    print(f"\n  {'Resultat':<22}{'Proba':>8}{'Cote':>8}{'EV':>9}")
    print("  " + "-" * 45)
    for k in ('a', 'draw', 'b'):
        print(f"  {label[k]:<22}{prob[k]:>7.1%}{cotes[k]:>8.0f}{ev[k]:>9.1f}")

    if mode == 'leader':
        choice, why = max(prob, key=prob.get), "couverture (proba max)"
    elif mode == 'challenger':
        best, underdog = max(ev, key=ev.get), min(prob, key=prob.get)
        if ev[underdog] >= 0.85 * ev[best] and prob[underdog] >= 0.12:
            choice, why = underdog, "variance assumee (EV proche, differenciant)"
        else:
            choice, why = best, "EV max"
    else:
        choice, why = max(ev, key=ev.get), "EV max"

    best_score, best_p = (0, 0), -1.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            ok = (choice == 'a' and i > j) or (choice == 'draw' and i == j) or (choice == 'b' and i < j)
            if ok and mat[i][j] > best_p:
                best_p, best_score = mat[i][j], (i, j)
    cells = sorted(((mat[i][j], i, j) for i in range(max_goals + 1) for j in range(max_goals + 1)),
                   reverse=True)[:3]

    print(f"\n  >> RESULTAT conseille ({mode}) : {label[choice]}  [{why}]")
    print(f"  >> SCORE a jouer : {best_score[0]}-{best_score[1]}  (P={best_p:.1%})")
    print(f"  Scores les plus probables : " +
          " | ".join(f"{i}-{j} ({p:.1%})" for p, i, j in cells))
    print("=" * 60)

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('team_a', nargs='?'); ap.add_argument('team_b', nargs='?')
    ap.add_argument('--cote-a', type=float); ap.add_argument('--cote-draw', type=float)
    ap.add_argument('--cote-b', type=float)
    ap.add_argument('--mode', choices=['equilibre', 'leader', 'challenger'], default='equilibre')
    ap.add_argument('--host', choices=['a', 'b'], default=None)
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()

    if a.demo:
        print("MODE DEMO (lambda fixes)")
        recommend("Equipe A", "Equipe B", 1.7, 0.9, {'a': 25, 'draw': 120, 'b': 280}, a.mode)
        return
    if not (a.team_a and a.team_b and a.cote_a and a.cote_draw and a.cote_b):
        print('Usage : ... "France" "Senegal" --cote-a 25 --cote-draw 120 --cote-b 280')
        print('Saisis les COTES REELLES de MPP. Lance d\'abord fit_team_ratings.py.')
        return
    if not (os.path.exists('data/team_ratings.json') or os.path.exists('data/processed_football_games.csv')):
        print("[STOP] Aucune donnee de forces. Lance fit_team_ratings.py.")
        return

    robj = load_ratings()
    if robj[0] == 'ema':
        print("[A VERIFIER] team_ratings.json absent -> forces EMA NON ajustees a l'adversaire")
        print("   (moins fiable, c'est ce qui sous-estimait le Mexique). Lance fit_team_ratings.py.")
    avail = teams_available(robj)
    for t in (a.team_a, a.team_b):
        if t not in avail:
            print(f"[STOP] '{t}' absent. Exemple de noms : {sorted(list(avail))[:8]}")
            return
    lam_a, lam_b = build_lambdas(a.team_a, a.team_b, robj, host=a.host)
    recommend(a.team_a, a.team_b, lam_a, lam_b,
              {'a': a.cote_a, 'draw': a.cote_draw, 'b': a.cote_b}, mode=a.mode)

if __name__ == "__main__":
    main()


