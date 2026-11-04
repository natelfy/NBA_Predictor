"""Planification des picks sous contrainte de verrou + règle d'arrêt pour le bonus x2.

Pourquoi planifier : prendre le meilleur joueur chaque soir brûle les stars sur des soirées moyennes alors
qu'elles seraient bloquées 31 jours. On maximise donc la somme des espérances sur tout l'horizon
(par défaut 31 jours = un cycle de verrou : chaque joueur est utilisable au plus une fois).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .core import LOCK_GAP_DAYS

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    HAVE_MILP = True
except ImportError:  # scipy < 1.9
    HAVE_MILP = False


def shrink_future(cand: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Lisse les EV des jours futurs vers la moyenne du joueur sur l'horizon (contre le biais du maximum :
    choisir le meilleur soir parmi 31 estimations bruitées sélectionne surtout du bruit). lam=1 : aucun lissage."""
    c = cand.copy()
    if lam >= 1.0:
        return c
    mean = c.groupby("PLAYER_ID")["EV"].transform("mean")
    fut = c["DAY"] > 0
    c.loc[fut, "EV"] = mean[fut] + lam * (c.loc[fut, "EV"] - mean[fut])
    return c


def available_from(last_picks: Dict[int, pd.Timestamp], today, gap: int = LOCK_GAP_DAYS) -> Dict[int, int]:
    """Premier jour (index relatif à today) où chaque joueur déjà pické redevient disponible."""
    today = pd.Timestamp(today)
    return {pid: max(0, (pd.Timestamp(d) + pd.Timedelta(days=gap) - today).days) for pid, d in last_picks.items()}


def _prepare(cand: pd.DataFrame, avail: Dict[int, int], per_day: int) -> pd.DataFrame:
    c = cand[["DAY", "PLAYER_ID", "EV"]].copy()
    c = c[c["DAY"] >= c["PLAYER_ID"].map(avail).fillna(0)]
    c = c.sort_values("EV", ascending=False).groupby("DAY").head(per_day)
    return c.reset_index(drop=True)


def _greedy(c: pd.DataFrame, gap: int) -> pd.DataFrame:
    chosen, last = [], {}
    for day, g in c.groupby("DAY"):
        for i, r in g.sort_values("EV", ascending=False).iterrows():
            if day - last.get(r.PLAYER_ID, -10 ** 6) >= gap:
                chosen.append(i)
                last[r.PLAYER_ID] = day
                break
    return c.loc[chosen]


def plan(cand: pd.DataFrame, avail: Optional[Dict[int, int]] = None, gap: int = LOCK_GAP_DAYS,
         per_day: int = 25, force: Optional[tuple] = None) -> pd.DataFrame:
    """cand : DAY (0 = aujourd'hui), PLAYER_ID, EV (une ligne par joueur et jour de match).
    Retourne les lignes choisies (un joueur par jour de match). force=(DAY, PLAYER_ID) impose un pick."""
    c = _prepare(cand, avail or {}, per_day)
    if force is not None:
        fd, fp = force
        if not ((c["DAY"] == fd) & (c["PLAYER_ID"] == fp)).any():
            extra = cand[(cand["DAY"] == fd) & (cand["PLAYER_ID"] == fp)][["DAY", "PLAYER_ID", "EV"]]
            c = pd.concat([c, extra]).reset_index(drop=True)
    if c.empty:
        return c
    if not HAVE_MILP:
        return _greedy(c, gap)

    n = len(c)
    days = np.sort(c["DAY"].unique())
    rows, cols = [], []
    lb, ub = [], []
    r = 0
    for d in days:  # exactement un pick par jour de match
        idx = np.flatnonzero(c["DAY"].to_numpy() == d)
        rows += [r] * len(idx); cols += list(idx); lb.append(1); ub.append(1); r += 1
    pids = c["PLAYER_ID"].to_numpy()
    dayv = c["DAY"].to_numpy()
    for pid in np.unique(pids):  # au plus un pick par fenêtre de `gap` jours et par joueur
        idx = np.flatnonzero(pids == pid)
        if len(idx) < 2:
            continue
        starts = [dayv.min()] if dayv.max() - dayv.min() < gap else range(int(dayv.min()), int(dayv.max()) + 1)
        for s in starts:
            w = idx[(dayv[idx] >= s) & (dayv[idx] < s + gap)]
            if len(w) > 1:
                rows += [r] * len(w); cols += list(w); lb.append(0); ub.append(1); r += 1
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, n))
    lo, hi = np.zeros(n), np.ones(n)
    if force is not None:
        m = (c["DAY"] == force[0]) & (c["PLAYER_ID"] == force[1])
        lo[m.to_numpy()] = 1
    res = milp(c=-c["EV"].to_numpy(), constraints=LinearConstraint(A, lb, ub), integrality=np.ones(n),
               bounds=Bounds(lo, hi), options={"time_limit": 20})
    if res.x is None:
        return _greedy(c, gap)
    return c[res.x > 0.5].sort_values("DAY")


def opportunity_costs(cand: pd.DataFrame, avail: Optional[Dict[int, int]] = None, gap: int = LOCK_GAP_DAYS,
                      top_k: int = 8, per_day: int = 25) -> pd.DataFrame:
    """Pour les top_k candidats du jour 0 : valeur totale du plan si on le force aujourd'hui,
    et manque à gagner vs le plan optimal (coût d'opportunité en points TTFL attendus)."""
    best = plan(cand, avail, gap, per_day)
    best_total = best["EV"].sum()
    today = _prepare(cand[cand["DAY"] == 0], avail or {}, top_k)
    out = []
    for _, r in today.iterrows():
        p = plan(cand, avail, gap, per_day, force=(0, r.PLAYER_ID))
        out.append({"PLAYER_ID": r.PLAYER_ID, "EV_TODAY": r.EV, "PLAN_TOTAL": p["EV"].sum(),
                    "REGRET": best_total - p["EV"].sum()})
    return pd.DataFrame(out).sort_values("REGRET")


# ----------------------------------------------------------------------------------
# Bonus x2 : arrêt optimal
# ----------------------------------------------------------------------------------
def x2_value_of_waiting(known_future_evs: list, n_unknown: int, samples: np.ndarray) -> float:
    """Espérance du gain x2 si on attend, en jouant optimalement ensuite.
    known_future_evs : EV planifiées des prochains jours de match connus (ordre chronologique).
    n_unknown : nombre de jours de match restants dans le mois APRÈS ces jours connus (EV tirée de `samples`)."""
    samples = np.asarray(samples, dtype=float)
    v = 0.0
    for _ in range(n_unknown):
        v = float(np.mean(np.maximum(samples, v))) if len(samples) else v
    for ev in reversed(known_future_evs):
        v = max(float(ev), v)
    return v


def x2_advice(today_ev: float, known_future_evs: list, n_unknown: int, samples: np.ndarray) -> dict:
    wait = x2_value_of_waiting(known_future_evs, n_unknown, samples)
    return {"use_today": bool(today_ev >= wait), "threshold": wait, "today_ev": float(today_ev)}
