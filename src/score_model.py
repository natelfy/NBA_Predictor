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


def default_bonus(i, j):
    """
    Bonus de score exact par defaut quand la vraie valeur MPP est inconnue.
    Hypothese : MPP paie ~ l'inverse de la popularite -> plus de buts = plus rare = plus de bonus,
    et les nuls 'larges' (2-2, 3-3) sont rares. Plafonne pour rester prudent.
    A REMPLACER par la vraie valeur MPP des qu'elle est connue (saisie par match).
    """
    base = 15 + 7 * (i + j)
    if i == j:
        base += 5
    return float(min(base, 120))


def exact_score_ev(score_m, bonus_known=None):
    """
    E[bonus de score exact] PAR issue + meilleur score (i,j) PAR issue.
    bonus_known : dict {(i,j): bonus_reel MPP} (sinon default_bonus).
    Le 'meilleur score' est cherche DANS chaque issue -> coherence issue<->score garantie.
    """
    bonus_known = bonus_known or {}
    eb = {'T1': 0.0, 'NUL': 0.0, 'T2': 0.0}
    best_score = {'T1': (1, 0), 'NUL': (1, 1), 'T2': (0, 1)}
    best_ev = {'T1': -1.0, 'NUL': -1.0, 'T2': -1.0}
    for (i, j), pr in score_m.items():
        bonus = bonus_known.get((i, j), default_bonus(i, j))
        ev = pr * bonus
        issue = 'T1' if i > j else ('NUL' if i == j else 'T2')
        eb[issue] += ev
        if ev > best_ev[issue]:
            best_ev[issue] = ev
            best_score[issue] = (i, j)
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
