"""
Football Oracle - Modele de score exact (Poisson bivarie + correction Dixon-Coles)
==================================================================================

Deux usages :
  1) Distribution P(score i-j) -> E[bonus de score exact] PAR issue (1N2).
  2) Probas d'issue 1N2 coherentes avec la matrice de score (jamais d'incoherence
     issue <-> score).

Source des buts attendus (lambda1, lambda2), par ordre de preference :
  a) team_ratings.json (Poisson ajuste a l'adversaire, fit_team_ratings.py)  <-- le mieux
  b) fallback : a partir d'un goal_diff predit + un total de buts attendu

Pur stdlib (math, json, os) : aucune dependance, testable hors-ligne.
ISOLATION : aucun fichier NBA.
"""
from math import exp, factorial
import json
import os

MAX_GOALS = 8          # grille de scores 0..MAX_GOALS
DEFAULT_RHO = -0.05    # correction Dixon-Coles (faible) sur les petits scores

# --- BONUS DE SCORE EXACT : BAREME OFFICIEL MPP (fonction en PALIERS, plafonnee) ---
# Le bonus depend de la PART des joueurs AYANT LE BON RESULTAT qui ont aussi le score exact
# (rarete du pronostic dans la foule), PAS d'une cote/proba continue :
#     part > 30%      -> +20   ("exact")
#     20 % - 30 %     -> +30   ("rare")
#     5 % - 20 %      -> +50   ("tres rare")
#     0,5 % - 5 %     -> +70   ("mega rare")
#     < 0,5 %         -> +100  ("ultra rare")
# On ne connait pas les picks exacts de la foule -> PROXY : la part est approchee par
# P(score | issue) du modele. [A VERIFIER] la foule est PLUS concentree que le modele sur
# les scores "evidents" (1-0, 1-1...), donc on SUR-estime un peu les paliers des scores rares.
# Plafond dur a +100 : viser plus rare qu'ultra-rare ne rapporte RIEN de plus.
BONUS_TIERS = [(0.30, 20.0), (0.20, 30.0), (0.05, 50.0), (0.005, 70.0)]   # (seuil part, bonus) ; sinon 100
BONUS_MAX = 100.0


def _pois(lmbda, k):
    return (lmbda ** k) * exp(-lmbda) / factorial(k)


def lambdas_from_ratings(team1, team2, ratings, is_t1_host=0):
    """Buts attendus via la regression de Poisson ajustee adversaire (team_ratings.json)."""
    inter = ratings['intercept']
    host = ratings.get('host_coef', 0.0)
    atk, dfn = ratings['attack'], ratings['defense']
    a1, d1 = atk.get(team1, 0.0), dfn.get(team1, 0.0)
    a2, d2 = atk.get(team2, 0.0), dfn.get(team2, 0.0)
    l1 = exp(inter + a1 + d2 + host * is_t1_host)
    l2 = exp(inter + a2 + d1)
    return max(0.05, l1), max(0.05, l2)


def lambdas_from_goal_diff(goal_diff, total_goals=2.6):
    """Fallback si pas de ratings : repartit un total de buts attendu selon l'ecart predit."""
    l1 = max(0.05, (total_goals + goal_diff) / 2.0)
    l2 = max(0.05, (total_goals - goal_diff) / 2.0)
    return l1, l2


def _dc_tau(i, j, l1, l2, rho):
    """Facteur de dependance Dixon-Coles (corrige Poisson independant sur 0-0/1-0/0-1/1-1)."""
    if i == 0 and j == 0:
        return 1.0 - l1 * l2 * rho
    if i == 0 and j == 1:
        return 1.0 + l1 * rho
    if i == 1 and j == 0:
        return 1.0 + l2 * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(l1, l2, rho=DEFAULT_RHO, max_goals=MAX_GOALS):
    """Matrice {(i,j): proba}, renormalisee a 1 (la troncature/DC ne casse pas la somme)."""
    m = {}
    s = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = _pois(l1, i) * _pois(l2, j) * _dc_tau(i, j, l1, l2, rho)
            if p < 0:
                p = 0.0
            m[(i, j)] = p
            s += p
    if s > 0:
        for k in m:
            m[k] /= s
    return m


def outcome_probs(score_m):
    """P(T1), P(NUL), P(T2) cohérentes avec la matrice de score."""
    p = {'T1': 0.0, 'NUL': 0.0, 'T2': 0.0}
    for (i, j), pr in score_m.items():
        if i > j:
            p['T1'] += pr
        elif i == j:
            p['NUL'] += pr
        else:
            p['T2'] += pr
    return p


def bonus_from_share(share):
    """Bareme MPP officiel : bonus selon la PART des joueurs (ayant le BON RESULTAT) qui ont
    le score exact. `share` in [0,1]. Fonction en paliers, plafonnee a +100."""
    for thr, pts in BONUS_TIERS:
        if share >= thr:
            return pts
    return BONUS_MAX


def bonus_map(score_m, bonus_known=None):
    """
    {score (i,j): bonus MPP} pour CE match. Bonus = bareme officiel applique a la part
    approchee P(score | issue) (proxy de la part de foule ayant le bon resultat).
    override ponctuel : bonus_known {score: bonus} si tu connais le vrai palier d'un score.
    """
    bk = dict(bonus_known) if bonus_known else {}
    p_issue = outcome_probs(score_m)
    out = {}
    for (i, j), pr in score_m.items():
        issue = 'T1' if i > j else ('NUL' if i == j else 'T2')
        denom = p_issue[issue]
        share = (pr / denom) if denom > 1e-12 else 0.0
        out[(i, j)] = bk.get((i, j), bonus_from_share(share))
    return out


def exact_score_ev(score_m, bonus_known=None):
    """
    Score le PLUS PROBABLE (modal) PAR issue + E[bonus] si tu nommes ce score.
    Bonus = bareme officiel MPP (bonus_map, paliers plafonnes a +100). On affiche le score
    MODAL (le plus probable) par issue -> lisible et coherent issue<->score.
    """
    bmap = bonus_map(score_m, bonus_known)
    best_score = {'T1': (1, 0), 'NUL': (1, 1), 'T2': (0, 1)}
    best_p = {'T1': -1.0, 'NUL': -1.0, 'T2': -1.0}
    for (i, j), pr in score_m.items():
        issue = 'T1' if i > j else ('NUL' if i == j else 'T2')
        if pr > best_p[issue]:
            best_p[issue] = pr
            best_score[issue] = (i, j)
    eb = {}
    for k, s in best_score.items():
        eb[k] = score_m.get(s, 0.0) * bmap.get(s, BONUS_MAX)   # E[bonus] en nommant le score modal
    return eb, best_score


def load_ratings(path='data/team_ratings.json'):
    """Charge team_ratings.json si present, sinon None (-> fallback goal_diff)."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------------------
# Auto-test (python src/score_model.py) : coherence, somme=1, scores par issue, blowout
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("== score_model :: auto-test ==")
    for label, (l1, l2) in [
        ("Portugal vs Uzbekistan (fort favori)", lambdas_from_goal_diff(1.8)),
        ("match equilibre", lambdas_from_goal_diff(0.0)),
        ("carton (Allemagne 7-1 type)", (3.4, 0.6)),
    ]:
        m = score_matrix(l1, l2)
        tot = sum(m.values())
        op = outcome_probs(m)
        eb, bs = exact_score_ev(m)
        print(f"\n{label}  lambda=({l1:.2f},{l2:.2f})  somme={tot:.4f}")
        print(f"  P(issue) : T1 {op['T1']:.1%} | NUL {op['NUL']:.1%} | T2 {op['T2']:.1%}")
        print(f"  meilleur score par issue : T1 {bs['T1']}  NUL {bs['NUL']}  T2 {bs['T2']}")
        assert abs(tot - 1.0) < 1e-9, "la matrice doit sommer a 1"
        assert bs['T1'][0] > bs['T1'][1] and bs['NUL'][0] == bs['NUL'][1] and bs['T2'][0] < bs['T2'][1]
    print("\nOK : sommes=1, issues coherentes avec les scores.")
