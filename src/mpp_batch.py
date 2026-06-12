"""
Football Oracle - Mode batch + placement du bonus x2 (MPP)  [v2 lisible + rarete]
================================================================================

Traite une journee de matchs et te dit, par match : QUOI jouer (selon ton mode),
le SCORE (modal + version "rarete" sous-cotee), et OU poser ton x2.

RAPPEL x2 : il est UNIQUE sur tout le tournoi. La ligne "BONUS x2" ci-dessous
designe le meilleur spot SI tu decides de le depenser aujourd'hui -- pas un x2
quotidien. Regle : jamais sur un favori, sur un pick qui sort du lot, et utilise-le.

SCORE "rarete" : on part du score le plus probable (Poisson) puis on penalise les
scores que les foules sur-jouent (1-0, 1-1, 2-0...). Sans le % par score de MPP,
c'est une HEURISTIQUE de jugement, pas un optimum chiffre -> a toi de trancher.

ENTREE CSV : team_a,team_b,cote_a,cote_draw,cote_b[,crowd_a,crowd_draw,crowd_b,host]
  -> cote_* = COTES DECIMALES reelles (1.83 / 3.65 / 4.70), PAS les points MPP.
Usage : python src/mpp_batch.py matchday.csv --mode challenger
"""
import os, csv, argparse
from mpp_match_engine import (load_ratings, teams_available, build_lambdas,
                              market_probs, score_matrix, result_probs)

# popularite "humaine" des scorelines (ce que la foule sur-joue). defaut 0.2
POP = {(1,0):1.0,(0,1):1.0,(1,1):1.0,(2,1):0.8,(1,2):0.8,(2,0):0.7,(0,2):0.7,
       (0,0):0.7,(3,1):0.45,(1,3):0.45,(3,0):0.4,(0,3):0.4,(2,2):0.4}
ALPHA = 0.70  # force du nudge anti-foule

def _ok(outcome, i, j):
    return (outcome=='a' and i>j) or (outcome=='draw' and i==j) or (outcome=='b' and i<j)

def modal_score(mat, outcome, mg):
    best, bp = (0,0), -1.0
    for i in range(mg+1):
        for j in range(mg+1):
            if _ok(outcome,i,j) and mat[i][j] > bp:
                bp, best = mat[i][j], (i,j)
    return best

def rarity_score(mat, outcome, mg):
    """score le + probable APRES penalisation des scorelines sur-jouees.
    Garde-fou : on n'accepte pas un score sous 50% de la proba du score modal."""
    modal_p = max(mat[i][j] for i in range(mg+1) for j in range(mg+1) if _ok(outcome,i,j))
    floor = 0.5 * modal_p
    best, bv = (0,0), -1.0
    for i in range(mg+1):
        for j in range(mg+1):
            if _ok(outcome,i,j) and mat[i][j] >= floor:
                adj = mat[i][j] * (1 - ALPHA*POP.get((i,j), 0.2))
                if adj > bv:
                    bv, best = adj, (i,j)
    return best

def top_scores(mat, outcome, mg, n=3):
    cells = [(mat[i][j], i, j) for i in range(mg+1) for j in range(mg+1) if _ok(outcome,i,j)]
    cells.sort(reverse=True)
    return [(i,j,p) for p,i,j in cells[:n]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--mode', choices=['equilibre','leader','challenger'], default='equilibre')
    a = ap.parse_args()
    if not os.path.exists(a.csv_path):
        print(f"[STOP] {a.csv_path} introuvable."); return
    ro = load_ratings(); av = teams_available(ro)
    with open(a.csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print("\n" + "="*70)
    print(f"  JOURNEE MPP  -  mode {a.mode}   ({len(rows)} matchs)")
    print("="*70)

    analyzed = []
    for r in rows:
        ta, tb = r['team_a'].strip(), r['team_b'].strip()
        if ta not in av or tb not in av:
            print(f"\n  {ta} vs {tb}  ->  [equipe absente des forces, ignore]"); continue
        cotes = {'a':float(r['cote_a']), 'draw':float(r['cote_draw']), 'b':float(r['cote_b'])}
        host = r.get('host','').strip() or None
        la, lb = build_lambdas(ta, tb, ro, host=host)
        mat = score_matrix(la, lb)
        p = market_probs(cotes)
        lab = {'a':ta, 'draw':'Nul', 'b':tb}
        fav = max(p, key=p.get)

        crowd = None
        if r.get('crowd_a') and r.get('crowd_draw') and r.get('crowd_b'):
            crowd = {'a':float(r['crowd_a'])/100,'draw':float(r['crowd_draw'])/100,'b':float(r['crowd_b'])/100}

        # quel PICK jouer selon le mode
        if a.mode == 'leader' or not crowd:
            pick = fav; why = "favori (proba max)"
            lev = None
        else:
            lev = {k: p[k]*(1-crowd[k]) for k in p}
            if a.mode == 'challenger':
                pick = max((k for k in lev if k != fav), key=lambda k: lev[k])
                why = f"differenciation (levier {lev[pick]:.2f})"
            else:  # equilibre
                pick = fav; why = "favori (EV max)"
        x2v = (lev[pick] if (lev and a.mode!='leader') else p[fav])

        analyzed.append(dict(ta=ta,tb=tb,host=host,p=p,lab=lab,fav=fav,pick=pick,
                             why=why,mat=mat,lev=lev,x2v=x2v,la=la,lb=lb,crowd=crowd))

    if not analyzed: return
    x2 = max(analyzed, key=lambda d: d['x2v'])

    for idx, d in enumerate(analyzed, 1):
        ms = modal_score(d['mat'], d['pick'], 8)
        rs = rarity_score(d['mat'], d['pick'], 8)
        tops = top_scores(d['mat'], d['pick'], 8)
        hote = f"  (hote: {d['host']})" if d['host'] else ""
        is_x2 = (d is x2)
        print(f"\n  Match {idx} : {d['ta']}  v  {d['tb']}{hote}")
        print(f"     buts attendus : {d['ta']} {d['la']:.2f} | {d['tb']} {d['lb']:.2f}")
        order = ['a','draw','b']
        probs = "  ".join(f"{d['lab'][k]} {d['p'][k]:.0%}"
                          + (f"/foule{int(d['crowd'][k]*100)}%" if d['crowd'] else "")
                          for k in order)
        print(f"     marche/foule  : {probs}")
        print(f"     >> JOUE       : {d['lab'][d['pick']]}   [{d['why']}]")
        print(f"     >> SCORE      : {ms[0]}-{ms[1]} (le + probable)"
              f"   |   RARETE: {rs[0]}-{rs[1]} (sous-coche, +gros bonus)")
        print(f"        autres scores : " + " ".join(f"{i}-{j}({pp:.0%})" for i,j,pp in tops))
        if is_x2:
            print(f"     ***  C'EST ICI que ton x2 a le + de valeur aujourd'hui  ***")

    print("\n" + "="*70)
    tgt = x2['pick']
    print(f"  x2 (si tu le depenses aujourd'hui) -> {x2['ta']} v {x2['tb']} : double {x2['lab'][tgt]}")
    print(f"  Rappel : x2 UNIQUE sur le tournoi. Jamais sur un favori. Utilise-le une fois.")
    if a.mode == 'challenger':
        print(f"  Mode chasseur : joue les picks differenciants ci-dessus pour te detacher.")
    print("="*70)

if __name__ == "__main__":
    main()

