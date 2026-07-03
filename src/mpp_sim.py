"""
Football Oracle - Simulateur P(top-3) a foule OBSERVEE (Monte-Carlo, stdlib pur)
================================================================================

Question reelle : "Quelle strategie de pronos maximise ma proba de finir top-3,
sachant que je suis a GAP points du 3e, dans une ligue ou je VOIS comment parie
la foule (% des joueurs par issue) ?"

Modele honnete "Y - L" (evite de simuler 600 joueurs) :
  - Y = points que TU gagnes sur les matchs a venir (selon TA strategie).
  - L = points que gagne la "ligne du 3e", modelisee par defaut comme un fort
        joueur qui suit la FOULE (le favori du public) sur chaque match.
  - Tu finis top-3  <=>  ton_total + Y >= total_3e + L  <=>  (Y - L) >= GAP.
  => P(top-3) ~= P(Y - L >= GAP).

Pourquoi c'est juste pour decider : quand tu joues comme la foule, Y - L ~ 0 ->
tu ne combles jamais un GAP positif. On ne grimpe qu'en ayant RAISON la ou la
foule a TORT (edge = proba_modele - %_foule > 0), ce qui paie la grosse cote.
Le simulateur classe donc les strategies par P(top-3) et te donne les picks.

Barreme MPP : points = cote de l'issue si correcte + bonus de score exact
(paliers officiels +20/30/50/70/100 selon rarete, score_model.bonus_map). x2 deja depense -> ignore.

Entrees fournies par l'utilisateur (il VOIT cotes, %-foule, et bonus connus) :
  match = {
    'name': 'Portugal vs Uzbekistan',
    'p_model': {'T1':.62,'NUL':.25,'T2':.13},   # proba du modele calibre
    'p_crowd': {'T1':.78,'NUL':.07,'T2':.15},   # % OBSERVE des joueurs (somme ~1)
    'cotes'  : {'T1':24,'NUL':154,'T2':173},    # cotes MPP
    'lambdas': (2.15, 0.35),                     # buts attendus (ratings ou goal_diff)
    'bonus_known': {(0,0):30, (1,1):15},         # bonus de score reels connus (optionnel)
  }

Pur stdlib (math, random, statistics) : testable hors-ligne, zero dependance.
ISOLATION : aucun fichier NBA.
"""
import random
import statistics
from score_model import (score_matrix, outcome_probs, exact_score_ev,
                         bonus_map, lambdas_from_goal_diff, lambdas_from_ratings)

OUTS = ['T1', 'NUL', 'T2']


# ----------------------------------------------------------------------------------
# Preparation d'un match : distribution de score, E[bonus]/issue, E[points]/issue...
# ----------------------------------------------------------------------------------
def prep_match(m):
    l1, l2 = m['lambdas']
    sm = score_matrix(l1, l2)
    ops = outcome_probs(sm)
    bonus_known = m.get('bonus_known') or {}
    eb, best_score = exact_score_ev(sm, bonus_known)
    bmap = bonus_map(sm, bonus_known)
    p_model, p_crowd, cotes = m['p_model'], m['p_crowd'], m['cotes']

    e_points = {k: p_model[k] * cotes[k] + eb[k] for k in OUTS}   # E[points] = issue + bonus
    nail, bonus_val = {}, {}
    for k in OUTS:
        s = best_score[k]
        po = ops[k] if ops[k] > 1e-9 else 1e-9
        nail[k] = min(1.0, sm.get(s, 0.0) / po)                  # P(score exact | issue k)
        bonus_val[k] = bmap.get(s, 0.0)                          # bonus MPP du score modal
    inv = {k: 1.0 / max(cotes[k], 1e-9) for k in OUTS}
    s_inv = sum(inv.values())
    p_imp = {k: inv[k] / s_inv for k in OUTS}                    # probas implicites des cotes
    return {
        'name': m.get('name', '?'),
        'p_model': p_model, 'p_crowd': p_crowd, 'cotes': cotes, 'p_imp': p_imp,
        'e_points': e_points, 'best_score': best_score,
        'nail': nail, 'bonus_val': bonus_val,
        'edge': {k: p_model[k] - p_crowd[k] for k in OUTS},      # avantage vs foule OBSERVEE
    }


# ----------------------------- strategies de picks --------------------------------
def picks_favori(P):
    """Ce que joue la foule / la ligne du 3e : l'issue la plus jouee par le public."""
    return [max(OUTS, key=lambda k: pm['p_crowd'][k]) for pm in P]


def picks_modele(P):
    return [max(OUTS, key=lambda k: pm['p_model'][k]) for pm in P]


EDGE_CAP = 0.22    # edge au-delà duquel on suspecte une ERREUR du modèle (pas une vraie value)
PROB_FLOOR = 0.12  # proba mini d'un pari (évite les longshots 6 % à grosse cote) ; garde les nuls (~25-30 %)
# --- LIGNE C (hybride discipliné) : on ne dégaine un coup contrarian QUE là où il a du levier ---
CROWD_STACK_MIN = 0.75   # ... si la foule s'EMPILE à >= 75 % sur une issue (vraie exposition)
EDGE_STRONG = 0.10       # ... OU si l'edge est FORT (>= +10 pts de proba)
MAX_CONTRARIAN = 2       # au plus 2 coups contrarian par journée ; favori partout ailleurs
# TRIANGULATION PAR LES COTES : un edge modèle > EDGE_CAP est normalement SUSPECT (erreur
# modèle probable)... SAUF si les probas implicites des cotes (bookmakers) confirment le
# modèle CONTRE la foule (p_implicite - p_foule >= BOOKIE_CONFIRM). 2 sources indépendantes
# d'accord contre la foule = biais de foule réel, pas une erreur (cas Australie-Égypte :
# cotes ~30/32/37 vs foule 13/25/62).
BOOKIE_CONFIRM = 0.10
# Back-test MODÈLE vs FOULE (PHASE DE POULES, 90') : le modèle ne battait la foule que sur
# le NUL -> doctrine "différenciation par le nul uniquement" (DIFF_ONLY_DRAW=True).
# CHANGEMENT DE RÉGIME (élimination directe, 120') : la prolongation ÉCRASE les nuls
# (~10-12 % réels) et transfère cette masse vers les VICTOIRES-surprises ; la foule a en
# plus appris à jouer le nul sur les matchs serrés (19-28 % vs 4-9 % en poules) -> l'edge
# "nul" est arbitré. La différenciation passe donc par l'issue rare la MOINS jouée, nul OU
# vainqueur contrarian, filtrée par les mêmes garde-fous (edge_min/EDGE_CAP/PROB_FLOOR).
# (Réversible : remettre True si on revenait à un régime 90'.)
DIFF_ONLY_DRAW = False


def _trustworthy(pm, k, edge_min, edge_cap, prob_floor):
    if pm['edge'][k] < edge_min or pm['p_model'][k] < prob_floor:
        return False
    # Edge énorme : suspect (erreur modèle)... sauf si les COTES confirment le modèle
    # contre la foule (2 sources indépendantes d'accord = biais de foule réel).
    if pm['edge'][k] > edge_cap and (pm['p_imp'][k] - pm['p_crowd'][k]) < BOOKIE_CONFIRM:
        return False
    crowd_fav = max(OUTS, key=lambda o: pm['p_crowd'][o])
    if DIFF_ONLY_DRAW and k != crowd_fav and k != 'NUL':
        return False   # pas de vainqueur contrarian : la foule nous bat sur le résultat
    # LIGNE C : sans EMPILEMENT de la foule ni edge FORT, un contrarian ne différencie
    # presque pas (l'issue est déjà couverte) et brûle de l'EV -> on suit la foule.
    if k != crowd_fav and pm['p_crowd'][crowd_fav] < CROWD_STACK_MIN and pm['edge'][k] < EDGE_STRONG:
        return False
    return True


def picks_valeur(P, edge_min=0.0, edge_cap=EDGE_CAP, prob_floor=PROB_FLOOR):
    """Meilleure E[points] parmi les issues à edge FIABLE ET proba >= plancher.
    Si rien de fiable -> on suit le FAVORI de la foule (on ne parie ni l'erreur, ni le longshot)."""
    out = []
    for pm in P:
        cand = [k for k in OUTS if _trustworthy(pm, k, edge_min, edge_cap, prob_floor)]
        if cand:
            out.append(max(cand, key=lambda k: pm['e_points'][k]))
        else:
            out.append(max(OUTS, key=lambda k: pm['p_crowd'][k]))   # repli prudent = favori foule
    return out


def picks_valeur_topK(P, K, edge_min=0.03, edge_cap=EDGE_CAP, prob_floor=PROB_FLOOR):
    """Différenciation sur les K matchs au plus fort edge FIABLE ; favori ailleurs.
    Exclut les edges énormes (erreur modèle) ET les longshots (proba < plancher)."""
    fav = picks_favori(P)
    val = picks_valeur(P, edge_min, edge_cap, prob_floor)
    diffs = sorted([(i, P[i]['edge'][val[i]]) for i in range(len(P))
                    if val[i] != fav[i] and _trustworthy(P[i], val[i], edge_min, edge_cap, prob_floor)],
                   key=lambda x: -x[1])
    keep = {i for i, _ in diffs[:K]}
    return [val[i] if i in keep else fav[i] for i in range(len(P))]


def sanitize(picks, P):
    """Filet de sécurité commun à TOUTES les stratégies : tout pari contrarian NON fiable
    (edge énorme = erreur modèle, longshot, sans levier ligne C) -> repli sur le favori de
    la foule ; puis discipline LIGNE C : au plus MAX_CONTRARIAN coups (les plus forts edges)."""
    fav = picks_favori(P)
    out = []
    for i, k in enumerate(picks):
        if k != fav[i] and not _trustworthy(P[i], k, 0.0, EDGE_CAP, PROB_FLOOR):
            out.append(fav[i])
        else:
            out.append(k)
    idx = [i for i in range(len(out)) if out[i] != fav[i]]
    if len(idx) > MAX_CONTRARIAN:
        keep = set(sorted(idx, key=lambda i: -P[i]['edge'][out[i]])[:MAX_CONTRARIAN])
        out = [out[i] if (out[i] == fav[i] or i in keep) else fav[i] for i in range(len(out))]
    return out


# ------------------------------- Monte-Carlo --------------------------------------
def _sample_outcome(p, rng):
    r = rng.random()
    c = 0.0
    for k in OUTS:
        c += p[k]
        if r <= c:
            return k
    return OUTS[-1]


def _match_points(pm, pick, outcome, rng):
    if pick != outcome:
        return 0.0
    pts = pm['cotes'][pick]
    if rng.random() < pm['nail'][pick]:      # tu trouves AUSSI le score exact -> bonus
        pts += pm['bonus_val'][pick]
    return pts


def simulate(P, your_picks, comp_picks, gap, trials=20000, seed=42):
    """Renvoie P(Y - L >= gap) + statistiques sur le gain net (Y - L)."""
    rng = random.Random(seed)
    fav = picks_favori(P)
    nets, hits_when_win, close = [], [], 0
    for _ in range(trials):
        Y = L = 0.0
        hits = 0
        for idx, pm in enumerate(P):
            o = _sample_outcome(pm['p_model'], rng)
            Y += _match_points(pm, your_picks[idx], o, rng)
            L += _match_points(pm, comp_picks[idx], o, rng)
            if your_picks[idx] != fav[idx] and your_picks[idx] == o:
                hits += 1
        net = Y - L
        nets.append(net)
        if net >= gap:
            close += 1
            hits_when_win.append(hits)
    nets.sort()
    return {
        'p_top3': close / trials,
        'net_median': nets[len(nets) // 2],
        'net_mean': sum(nets) / len(nets),
        'net_p90': nets[min(len(nets) - 1, int(0.90 * len(nets)))],
        'avg_contrarian_hits_when_win': (sum(hits_when_win) / len(hits_when_win)) if hits_when_win else 0.0,
    }


# ------------------------- API haut niveau : recommander --------------------------
def recommend(matches, gap, trials=20000, seed=42, competitor='favori', objective='target'):
    """
    Compare les strategies et renvoie la meilleure + les picks recommandes.
    competitor : strategie supposee de la 'ligne du 3e' ('favori' par defaut ;
                 'valeur' = concurrent plus agressif = scenario prudent).
    objective  : 'target' -> maximise P(atteindre l'objectif du jour) (defaut, sain) ;
                 'ceiling' -> maximise le PLAFOND (net p90) = mode chasse fin de tournoi.
    """
    P = [prep_match(m) for m in matches]
    comp = picks_valeur(P) if competitor == 'valeur' else picks_favori(P)

    raw_cands = {
        'favori (comme la foule)': picks_favori(P),
        'modele (argmax proba)': picks_modele(P),
        'valeur (edge>0)': picks_valeur(P, edge_min=0.0),
        'valeur top-3 edges': picks_valeur_topK(P, 3),
    }
    cands = {name: sanitize(pk, P) for name, pk in raw_cands.items()}   # filet commun anti-mauvais-pari
    table = {}
    for name, yp in cands.items():
        table[name] = simulate(P, yp, comp, gap, trials, seed)

    key = 'net_p90' if objective == 'ceiling' else 'p_top3'
    best_name = max(table, key=lambda n: table[n][key])
    best_picks = cands[best_name]
    fav = picks_favori(P)
    detail = []
    for idx, pm in enumerate(P):
        k = best_picks[idx]
        detail.append({
            'match': pm['name'], 'pick': k, 'score': pm['best_score'][k],
            'p_model': pm['p_model'][k], 'p_crowd': pm['p_crowd'][k],
            'edge': pm['edge'][k], 'cote': pm['cotes'][k], 'p_imp': pm['p_imp'][k],
            'contrarian': k != fav[idx],
        })
    return {'best': best_name, 'p_top3': table[best_name]['p_top3'],
            'table': table, 'picks': detail, 'P': P}


def build_match(name, p_model, p_crowd, cotes, goal_diff=None, lambdas=None,
                ratings=None, teams=None, bonus_known=None, is_t1_host=0):
    """Helper : construit un 'match' en derivant lambdas des ratings, sinon du goal_diff."""
    if lambdas is None:
        if ratings is not None and teams is not None:
            lambdas = lambdas_from_ratings(teams[0], teams[1], ratings, is_t1_host)
        else:
            lambdas = lambdas_from_goal_diff(goal_diff if goal_diff is not None else 0.0)
    return {'name': name, 'p_model': p_model, 'p_crowd': p_crowd, 'cotes': cotes,
            'lambdas': lambdas, 'bonus_known': bonus_known or {}}


# --------------------------------------------------------------------------------------
# Auto-test / demo (python src/mpp_sim.py) : verifie la mecanique du moteur
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Slate synthetique : sur quelques matchs, le modele voit des nuls/outsiders
    # que la foule sous-estime (edge>0) -> ce sont les coups qui font remonter.
    slate = [
        build_match("Favori net A", {'T1':.70,'NUL':.22,'T2':.08},
                    {'T1':.82,'NUL':.10,'T2':.08}, {'T1':30,'NUL':110,'T2':160}, goal_diff=1.4),
        build_match("Favori net B", {'T1':.66,'NUL':.24,'T2':.10},
                    {'T1':.80,'NUL':.12,'T2':.08}, {'T1':33,'NUL':120,'T2':150}, goal_diff=1.2),
        build_match("Nul sous-cote", {'T1':.40,'NUL':.34,'T2':.26},
                    {'T1':.55,'NUL':.12,'T2':.33}, {'T1':70,'NUL':150,'T2':95}, goal_diff=0.2),
        build_match("Outsider live", {'T1':.34,'NUL':.30,'T2':.36},
                    {'T1':.50,'NUL':.20,'T2':.30}, {'T1':95,'NUL':140,'T2':120}, goal_diff=-0.1),
        build_match("Equilibre", {'T1':.38,'NUL':.32,'T2':.30},
                    {'T1':.45,'NUL':.18,'T2':.37}, {'T1':80,'NUL':150,'T2':100}, goal_diff=0.1),
        build_match("Piege", {'T1':.30,'NUL':.33,'T2':.37},
                    {'T1':.40,'NUL':.22,'T2':.38}, {'T1':110,'NUL':135,'T2':95}, goal_diff=-0.2),
    ]
    P = [prep_match(m) for m in slate]

    print("== mpp_sim :: auto-test (mecanique) ==")
    fav = picks_favori(P)
    # 1) Jouer comme la foule -> gain net ~ 0 -> on ne comble pas un gap positif
    s_fav = simulate(P, fav, fav, gap=120, trials=8000, seed=1)
    print(f"favori vs favori : net_mean={s_fav['net_mean']:.1f} (attendu ~0), "
          f"P(combler 120)={s_fav['p_top3']:.1%}")
    assert abs(s_fav['net_mean']) < 6.0, "jouer comme la foule doit donner un net ~0"

    # 2) Jouer la valeur (edge>0) -> net moyen > 0 et P(combler) > favori
    val = picks_valeur(P)
    s_val = simulate(P, val, fav, gap=120, trials=8000, seed=1)
    print(f"valeur vs favori : net_mean={s_val['net_mean']:.1f} (attendu >0), "
          f"P(combler 120)={s_val['p_top3']:.1%}")
    assert s_val['net_mean'] > s_fav['net_mean'], "la valeur doit battre le favori en moyenne"
    assert s_val['p_top3'] >= s_fav['p_top3'], "la valeur doit augmenter P(combler)"

    print("\n== Recommandation (gap au 3e = 120 sur ce slate) ==")
    rec = recommend(slate, gap=120, trials=12000)
    for name, r in rec['table'].items():
        print(f"  {name:<26} P(top3)={r['p_top3']:.1%}  net_med={r['net_median']:+.0f}  "
              f"net_p90={r['net_p90']:+.0f}")
    print(f"\n  >>> MEILLEURE : {rec['best']}  (P(top3)={rec['p_top3']:.1%})")
    for d in rec['picks']:
        flag = " <- CONTRARIAN" if d['contrarian'] else ""
        print(f"     {d['match']:<16} {d['pick']:<4} {d['score'][0]}-{d['score'][1]}  "
              f"modele {d['p_model']:.0%} vs foule {d['p_crowd']:.0%} (edge {d['edge']*100:+.0f}) "
              f"cote {d['cote']}{flag}")
    print("\nOK : favori->net~0, valeur->net>0 et P(top3) superieure. Mecanique validee.")
