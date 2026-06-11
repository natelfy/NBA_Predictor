"""
Football Oracle - Mode batch + placement du bonus x2 (MPP)
==========================================================

Traite une journee de matchs d'un coup et te dit OU poser ton bonus x2.

VERITE : EV plate -> doubler n'importe quel match ajoute la meme esperance.
Le x2 est un AMPLIFICATEUR DE VARIANCE, pas un generateur de points. On le
place donc selon ta position en ligue :
  - leader      : x2 sur le favori le plus sur (amplifier avec le peloton).
  - challenger  : x2 sur le meilleur levier de differenciation (bombe de rattrapage).
  - equilibre   : x2 sur le meilleur coup de differenciation (un swing calcule).

[A VERIFIER] On ne sait pas ou les ADVERSAIRES posent leur x2 (donnee non dispo).
On optimise donc TA strategie, pas la guerre des x2.

ENTREE : un CSV (team_a,team_b,cote_a,cote_draw,cote_b[,crowd_a,crowd_draw,crowd_b,host])
Usage : python src/mpp_batch.py matches.csv --mode challenger
"""
import os, sys, csv, argparse
from mpp_match_engine import (load_ratings, teams_available, build_lambdas,
                              market_probs, score_matrix, result_probs)

def modal_score(mat, outcome, mg):
    best, bp = (0, 0), -1.0
    for i in range(mg + 1):
        for j in range(mg + 1):
            ok = (outcome == 'a' and i > j) or (outcome == 'draw' and i == j) or (outcome == 'b' and i < j)
            if ok and mat[i][j] > bp:
                bp, best = mat[i][j], (i, j)
    return f"{best[0]}-{best[1]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--mode', choices=['equilibre', 'leader', 'challenger'], default='equilibre')
    a = ap.parse_args()
    if not os.path.exists(a.csv_path):
        print(f"[STOP] {a.csv_path} introuvable."); return
    ro = load_ratings(); av = teams_available(ro)

    rows = []
    with open(a.csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    print("\n" + "=" * 78)
    print(f"  JOURNEE MPP - mode {a.mode}   ({len(rows)} matchs)")
    print("=" * 78)
    print(f"  {'Match':<26}{'Favori':<14}{'Diff (levier)':<20}{'x2?':>6}")
    print("  " + "-" * 70)

    analyzed = []
    for r in rows:
        ta, tb = r['team_a'].strip(), r['team_b'].strip()
        if ta not in av or tb not in av:
            print(f"  {ta} vs {tb:<14} [equipe absente, ignore]"); continue
        cotes = {'a': float(r['cote_a']), 'draw': float(r['cote_draw']), 'b': float(r['cote_b'])}
        host = r.get('host', '').strip() or None
        la, lb = build_lambdas(ta, tb, ro, host=host)
        mat = score_matrix(la, lb)
        p_mkt = market_probs(cotes)
        lab = {'a': ta, 'draw': 'Nul', 'b': tb}
        fav = max(p_mkt, key=p_mkt.get)
        fav_str = f"{lab[fav]} {modal_score(mat, fav, 8)}"

        crowd = None
        if r.get('crowd_a') and r.get('crowd_draw') and r.get('crowd_b'):
            crowd = {'a': float(r['crowd_a'])/100, 'draw': float(r['crowd_draw'])/100, 'b': float(r['crowd_b'])/100}
        if crowd:
            lev = {k: p_mkt[k] * (1 - crowd[k]) for k in p_mkt}
            diff = max((k for k in lev if k != fav), key=lambda k: lev[k])
            diff_str = f"{lab[diff]} {modal_score(mat, diff, 8)} ({lev[diff]:.2f})"
            x2_score = lev[diff] if a.mode != 'leader' else p_mkt[fav]
        else:
            lev = None; diff = None; diff_str = "(pas de % foule)"; x2_score = p_mkt[fav]

        analyzed.append({'m': f"{ta} v {tb}", 'fav': fav_str, 'diff': diff_str,
                         'x2_score': x2_score, 'lev': lev, 'fav_k': fav, 'diff_k': diff, 'lab': lab})

    # placement du x2
    if analyzed:
        x2 = max(analyzed, key=lambda d: d['x2_score'])
        for d in analyzed:
            mark = "  <== x2" if d is x2 else ""
            print(f"  {d['m']:<26}{d['fav']:<14}{d['diff']:<20}{mark:>6}")
        print("\n" + "=" * 78)
        if a.mode == 'leader':
            print(f"  BONUS x2 -> {x2['m']} : double le FAVORI {x2['lab'][x2['fav_k']]} "
                  f"(le plus sur, tu restes avec le peloton)")
        else:
            tgt = x2['diff_k'] if x2['diff_k'] else x2['fav_k']
            print(f"  BONUS x2 -> {x2['m']} : double {x2['lab'][tgt]} "
                  f"(meilleur levier de differenciation de la journee)")
        print("  Sur les AUTRES matchs : joue les favoris (reste avec le peloton).")
        print("=" * 78)

if __name__ == "__main__":
    main()

