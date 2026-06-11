"""
Football Oracle - Simulateur Monte-Carlo du jeu MPP (vectorise)
===============================================================

Tranche : "quelle strategie maximise P(finir 1er) dans ma ligue ?"
Hypothese honnete : meme info pour tous (les cotes). Seul l'ecart = la STRATEGIE.

ETALON 'champ_moyen' = jouer comme un adversaire -> DOIT ~= hasard 1/(N+1).
C'est le test de correction du moteur.

Strategies testees :
  champ_moyen : joue comme la foule (etalon).
  favori      : favori partout + x2 sur le + sur (EV-optimal, variance mini).
  diff        : 1 swing sur le match le + ouvert + x2 dessus (differenciation douce).
  longshot    : outsiders (grosse cote) sur les K matchs les + ouverts + x2 sur
                le + gros outsider (variance MAX = theorie des concours).
  adaptatif   : favori si tu mene, longshot si tu chasses.

[A VERIFIER] Bareme : points ~ cote ; score exact -> double le match ; x2 -> double
le match. Bonus de RARETE non modelise (Famille A). On isole le levier issue+x2.
"""
import numpy as np
import argparse

SCORES_WIN  = [(1,0),(2,0),(2,1),(3,0),(3,1)]
SCORES_DRAW = [(1,1),(0,0),(2,2)]
POP_WIN  = np.array([.36,.26,.20,.11,.07]); POP_WIN /= POP_WIN.sum()
POP_DRAW = np.array([.55,.30,.15])
# codes de score par issue (0=dom,1=nul,2=ext) ; ext = miroir de dom
CODES = {0: np.array([10*a+b for a,b in SCORES_WIN]),
         1: np.array([10*a+b for a,b in SCORES_DRAW]),
         2: np.array([10*b+a for a,b in SCORES_WIN])}
POP   = {0: POP_WIN, 1: POP_DRAW, 2: POP_WIN}

def gen_probs(rng):
    pf = rng.uniform(0.38, 0.84); rem = 1 - pf
    pd = rem * rng.uniform(0.40, 0.62)
    p = np.array([pf, pd, rem - pd])
    return p[rng.permutation(3)]

def cotes_from(p, margin=0.06): return 1.0 / (p * (1 + margin))
def res_pts(c): return c * 10.0

def sample_codes(outs, rng, pop_prob=0.8):
    """Vectorise : un code de score par joueur selon son issue."""
    n = len(outs); codes = np.empty(n, dtype=int)
    pop_flag = rng.random(n) < pop_prob
    for o in (0, 1, 2):
        m = outs == o
        k = int(m.sum())
        if not k: continue
        cds = CODES[o]; pf = pop_flag[m]; idx = np.empty(k, dtype=int)
        npop = int(pf.sum())
        if npop: idx[pf] = rng.choice(len(cds), size=npop, p=POP[o])
        if k - npop: idx[~pf] = rng.integers(len(cds), size=k - npop)
        codes[m] = cds[idx]
    return codes

def one_code(out, rng, popular=True):
    cds = CODES[out]
    i = rng.choice(len(cds), p=POP[out]) if popular else rng.integers(len(cds))
    return int(cds[i])

def my_day(cotes, strat, rng, leading, K=3, pfav=0.80, popp=0.8):
    G = len(cotes); favs = [int(np.argmin(c)) for c in cotes]
    open_cote = np.array([cotes[g][favs[g]] for g in range(G)])
    most_open = int(open_cote.argmax()); safest = int(open_cote.argmin())
    longs = [int(np.argmax(cotes[g])) for g in range(G)]          # issue + cotee
    long_pts = np.array([cotes[g][longs[g]] for g in range(G)])
    if strat == 'adaptatif': strat = 'favori' if leading else 'longshot'

    outs = list(favs); codes = []; x2 = safest
    if strat == 'champ_moyen':
        for g in range(G):
            c = cotes[g]
            if rng.random() < pfav: o = favs[g]
            else:
                w = 1/c; w[favs[g]] = 0; w /= w.sum(); o = int(rng.choice(3, p=w))
            outs[g] = o; codes.append(one_code(o, rng, rng.random() < popp))
        return np.array(outs), np.array(codes), safest
    if strat == 'favori':
        codes = [one_code(favs[g], rng, True) for g in range(G)]
        return np.array(favs), np.array(codes), safest
    if strat == 'diff':
        for g in range(G):
            if g == most_open:
                c = cotes[g]; w = 1/c; w[favs[g]] = 0; w /= w.sum()
                o = int(w.argmax()); outs[g] = o
                codes.append(one_code(o, rng, popular=False))
            else: codes.append(one_code(favs[g], rng, True))
        return np.array(outs), np.array(codes), most_open
    if strat == 'longshot':
        openK = set(np.argsort(open_cote)[-K:])                   # K matchs + ouverts
        for g in range(G):
            if g in openK:
                outs[g] = longs[g]; codes.append(one_code(longs[g], rng, popular=False))
            else: codes.append(one_code(favs[g], rng, True))
        x2 = int(long_pts.argmax())                               # x2 sur + gros outsider
        return np.array(outs), np.array(codes), x2
    raise ValueError(strat)

def run(strat, n, T, days, gpd, seed, K=3, pfav=0.80, popp=0.8):
    rng = np.random.default_rng(seed)
    ranks = np.empty(T); firsts = 0
    for t in range(T):
        my = 0.0; fld = np.zeros(n)
        for d in range(days):
            probs = [gen_probs(rng) for _ in range(gpd)]
            cotes = [cotes_from(p) for p in probs]
            res = [(int(rng.choice(3, p=p)), None) for p in probs]
            res = [(o, one_code(o, rng, rng.random() < 0.5)) for o, _ in res]
            favs = [int(np.argmin(c)) for c in cotes]
            fav_cote = np.array([cotes[g][favs[g]] for g in range(gpd)])
            fx2 = int(fav_cote.argmin())                          # foule : x2 sur + sur
            leading = (d > 0 and my >= fld.max())
            mo, mc, mx2 = my_day(cotes, strat, rng, leading, K, pfav, popp)
            for g in range(gpd):
                c = cotes[g]; to, tcode = res[g]; base = res_pts(c[to])
                # --- champ (vectorise sur n) ---
                fav = favs[g]; fo = np.full(n, fav)
                u = rng.random(n); nf = u >= pfav
                if nf.any():
                    w = 1/c.copy(); w[fav] = 0; w /= w.sum()
                    fo[nf] = rng.choice(3, size=int(nf.sum()), p=w)
                fc = sample_codes(fo, rng, popp)
                correct = fo == to
                pts = correct * base + (correct & (fc == tcode)) * base
                if g == fx2: pts = pts * 2
                fld += pts
                # --- toi ---
                if mo[g] == to:
                    mp = base + (base if mc[g] == tcode else 0)
                    my += mp * 2 if g == mx2 else mp
        ranks[t] = 1 + int((fld > my).sum())
        if my >= fld.max(): firsts += 1
    return ranks.mean(), firsts / T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tournois', type=int, default=4000)
    ap.add_argument('--joueurs', type=int, default=20)
    ap.add_argument('--jours', type=int, default=6)
    ap.add_argument('--matchs', type=int, default=8)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pfav', type=float, default=0.80)
    ap.add_argument('--popp', type=float, default=0.80)
    a = ap.parse_args()
    base = 1.0 / (a.joueurs + 1)
    print("\n" + "=" * 74)
    print(f"  SIMULATEUR MPP  |  {a.tournois} tournois, {a.joueurs+1} joueurs, "
          f"{a.jours}j x {a.matchs} matchs  (K={a.K}, pfav={a.pfav})")
    print(f"  Meme info pour tous -> seul l'ecart = la STRATEGIE.")
    print("=" * 74)
    print(f"  {'Strategie':<16}{'Rang moyen':>12}{'P(1er)':>11}{'vs hasard':>13}")
    print("  " + "-" * 52)
    for s in ['champ_moyen', 'favori', 'diff', 'longshot', 'adaptatif']:
        mr, p1 = run(s, a.joueurs, a.tournois, a.jours, a.matchs, a.seed, a.K, a.pfav, a.popp)
        tag = "  <-- etalon" if s == 'champ_moyen' else ""
        print(f"  {s:<16}{mr:>12.2f}{p1:>10.1%}{p1/base:>12.2f}x{tag}")
    print("  " + "-" * 52)
    print(f"  hasard pur : P(1er) {base:.1%}, rang moyen {(a.joueurs+2)/2:.1f}")
    print("=" * 74)

if __name__ == "__main__":
    main()


