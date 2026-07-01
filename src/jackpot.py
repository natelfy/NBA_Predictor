"""
Football Oracle - Mode JACKPOT (scores exacts) : le baroud d'honneur
====================================================================

Quand l'ecart est ENORME, le seul levier avec assez d'upside pour remonter, c'est
NAILER des SCORES EXACTS (cote de l'issue + BONUS de rarete), comme la joueuse qui
a pris la 1re place en devinant 3 scores exacts.

Le BONUS suit le BAREME OFFICIEL MPP : une fonction en PALIERS (+20/+30/+50/+70/+100)
selon la PART des joueurs (ayant le bon resultat) qui ont le score exact -- PLAFONNEE a +100.
Proxy de cette part : P(score | issue) du modele. Deux consequences clefs :
  1) le bonus (<=100) est PETIT face a la cote d'une issue rare (nul/upset a 120-180) -> le
     vrai levier reste de trouver le bon RESULTAT rare ; le score exact n'est qu'un bonus.
  2) viser plus rare qu'"ultra rare" (<0,5%) ne rapporte RIEN de plus (plafond) -> inutile
     de sacrifier de la proba d'occurrence pour un score absurde.
Le module compare des profils (score probable -> score rare plausible) et le simulateur
choisit celui qui maximise P(top-3) vs le champ (favori + score populaire).

⚠️ Honnetete : a tres gros deficit, meme optimal, c'est une LOTERIE. Ce module
maximise une (petite) proba, il ne la cree pas.

Pur stdlib. Bonus = bareme MPP (bonus_map) ; override ponctuel possible. ISOLATION : aucun fichier NBA.
"""
import random
from score_model import (score_matrix, bonus_map,
                         lambdas_from_goal_diff, lambdas_from_ratings)

OUTS = ['T1', 'NUL', 'T2']


def _issue(i, j):
    return 'T1' if i > j else ('NUL' if i == j else 'T2')


def score_table(lambdas, cotes, bonus_known=None, max_goals=6):
    """Par score (i,j) : P, issue, cote, bonus (bareme MPP officiel), reward=cote+bonus, ev=P*reward.
    bonus = palier selon la part P(score|issue) (proxy foule) ; override par bonus_known."""
    sm = score_matrix(lambdas[0], lambdas[1])
    bmap = bonus_map(sm, bonus_known)
    rows = []
    for (i, j), p in sm.items():
        if i > max_goals or j > max_goals:
            continue
        iss = _issue(i, j)
        b = bmap[(i, j)]
        rows.append({'score': (i, j), 'p': p, 'issue': iss, 'cote': cotes[iss],
                     'bonus': b, 'reward': cotes[iss] + b, 'ev': p * (cotes[iss] + b)})
    rows.sort(key=lambda r: -r['ev'])
    return rows


def _prep(match):
    tbl = score_table(match['lambdas'], match['cotes'], match.get('bonus_known'))
    sm = score_matrix(match['lambdas'][0], match['lambdas'][1])
    items = sorted(sm.items(), key=lambda kv: kv[0])
    bonus_of = bonus_map(sm, match.get('bonus_known'))               # bonus par score = bareme MPP
    fav = min(OUTS, key=lambda k: match['cotes'][k])            # favori foule ~ plus petite cote
    fav_scores = [r for r in tbl if r['issue'] == fav]
    field_score = max(fav_scores, key=lambda r: r['p'])['score'] if fav_scores else (1, 0)
    return {'name': match.get('name', '?'), 'cotes': match['cotes'], 'table': tbl,
            'items': items, 'fav': fav, 'field_score': field_score, 'bonus_of': bonus_of}


def _pick_ev(prep):
    """Score a meilleure esperance de points (P x (cote+bonus)) -> souvent le score probable."""
    return prep['table'][0]['score']


def _pick_upside(prep, p_floor=0.05):
    """Score le PLUS PAYANT (= le plus rare) parmi ceux encore plausibles (P >= p_floor)."""
    cand = [r for r in prep['table'] if r['p'] >= p_floor]
    return (max(cand, key=lambda r: r['reward'])['score'] if cand else prep['table'][0]['score'])


def _sample(items, rng):
    r = rng.random(); c = 0.0
    for ij, p in items:
        c += p
        if r <= c:
            return ij
    return items[-1][0]


def simulate(preps, your_scores, gap, trials=20000, seed=42):
    rng = random.Random(seed)
    nets = []
    wins = 0
    for _ in range(trials):
        net = 0.0
        for k, pm in enumerate(preps):
            ti, tj = _sample(pm['items'], rng)
            tiss = _issue(ti, tj)
            ys = your_scores[k]
            yiss = _issue(ys[0], ys[1])
            you = pm['cotes'][yiss] * (yiss == tiss) + pm['bonus_of'].get(ys, 0.0) * (ys == (ti, tj))
            fs = pm['field_score']
            fld = pm['cotes'][pm['fav']] * (pm['fav'] == tiss) + pm['bonus_of'].get(fs, 0.0) * (fs == (ti, tj))
            net += you - fld
        nets.append(net)
        if net >= gap:
            wins += 1
    nets.sort()
    return {'p_top3': wins / trials, 'net_median': nets[len(nets) // 2],
            'net_p90': nets[min(len(nets) - 1, int(0.9 * len(nets)))], 'net_max': nets[-1]}


def recommend(matches, gap, trials=20000, seed=42):
    preps = [_prep(m) for m in matches]
    # Bonus plafonne (<=100) -> il ne "cree" pas la remontada, c'est la cote de l'issue rare qui
    # le fait. On propose du plus sur (score probable) au plus rare-plausible, et le simulateur
    # tranche sur P(top-3) selon la taille de l'ecart (gros ecart -> plus de variance).
    strategies = {
        'sûr (score probable)':               [_pick_ev(p) for p in preps],
        'upside (rare plausible)':            [_pick_upside(p, 0.05) for p in preps],
        'loterie (très rare · variance max)': [_pick_upside(p, 0.02) for p in preps],
    }
    table = {name: simulate(preps, sc, gap, trials, seed) for name, sc in strategies.items()}
    table['champ (favori+score populaire)'] = simulate(preps, [p['field_score'] for p in preps], gap, trials, seed)
    best = max(strategies, key=lambda n: table[n]['p_top3'])
    your = strategies[best]
    picks = []
    for k, p in enumerate(preps):
        s = your[k]
        row = next(r for r in p['table'] if r['score'] == s)
        picks.append({'match': p['name'], 'score': s, 'issue': row['issue'], 'p': row['p'],
                      'cote': row['cote'], 'bonus': row['bonus'], 'ev': row['ev'],
                      'alts': p['table'][:5]})
    return {'best': best, 'p_top3': table[best]['p_top3'], 'net_p90': table[best]['net_p90'],
            'net_max': table[best]['net_max'], 'table': table, 'picks': picks}


def build_match(name, cotes, goal_diff=None, lambdas=None, ratings=None, teams=None,
                bonus_known=None, is_t1_host=0):
    if lambdas is None:
        if ratings is not None and teams is not None:
            lambdas = lambdas_from_ratings(teams[0], teams[1], ratings, is_t1_host)
        else:
            lambdas = lambdas_from_goal_diff(goal_diff if goal_diff is not None else 0.0)
    return {'name': name, 'cotes': cotes, 'lambdas': lambdas, 'bonus_known': bonus_known or {}}


# ------------------------------------ auto-test ------------------------------------
if __name__ == "__main__":
    # Grille du jour (cotes réelles ; λ approx via goal_diff issu des probas modèle)
    slate = [
        build_match("England vs DR Congo", {'T1': 10, 'NUL': 144, 'T2': 250}, goal_diff=1.7),
        build_match("Belgium vs Senegal", {'T1': 61, 'NUL': 100, 'T2': 123}, goal_diff=0.7),
        build_match("USA vs Bosnia", {'T1': 12, 'NUL': 132, 'T2': 200}, goal_diff=1.4),
    ]
    rec = recommend(slate, gap=117, trials=15000)
    print("== JACKPOT :: grille du jour (bonus = rarete du score dans le match) ==")
    for n, r in rec['table'].items():
        print(f"  {n:<34} P(top3)={r['p_top3']:.1%}  net_p90={r['net_p90']:+.0f}  net_max={r['net_max']:+.0f}")
    print(f"\n  >>> MEILLEUR PROFIL : {rec['best']}  (P(top3)={rec['p_top3']:.1%})")
    for d in rec['picks']:
        i, j = d['score']
        print(f"     {d['match']:<22} -> {i}-{j} ({d['issue']}) | P={d['p']:.1%} "
              f"cote {d['cote']} + bonus {d['bonus']:.0f} = {d['cote']+d['bonus']:.0f} pts si exact")
        alts = ", ".join(f"{r['score'][0]}-{r['score'][1]}(P{r['p']*100:.0f}%·{r['cote']+r['bonus']:.0f}pts)"
                         for r in d['alts'])
        print(f"        alternatives : {alts}")
