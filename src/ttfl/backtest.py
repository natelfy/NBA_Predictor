"""Backtest d'une saison TTFL complète avec verrou de 31 jours.

Compare, sur la saison de validation (le modèle n'est entraîné que sur les saisons précédentes) :
  - greedy_l10   : chaque soir, le joueur dispo avec la meilleure moyenne sur 10 matchs (ce que fait l'Advisor TTFL)
  - greedy_model : chaque soir, la meilleure espérance du modèle (EV = P_PLAY x MU)
  - planner      : plan optimal sur 31 jours (MILP), on exécute le pick du jour, on re-planifie chaque soir

Réalisme : quand un joueur de rotation ne joue pas, avec proba --known-out (0.8) son absence était connue avant la
fermeture du deck (il n'est pas pickable) ; sinon c'est une absence tardive : 0 point + verrou (ce qui arrive en vrai).

Biais connus (à garder en tête en lisant les chiffres) :
  - les pronostics des jours futurs de l'horizon utilisent l'état connu à la date du pick (pas de fuite),
    mais ignorent les blessures futures (comme en vrai) ;
  - la distribution d'EV pour la règle x2 est estimée sur la saison testée (elle ne fuit aucun résultat).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .core import LOCK_GAP_DAYS, X2_MONTHS
from .features import FeatureBuilder, prepare_games
from .model import TTFLModel, build_tables, holdout_season
from .planner import available_from, plan, shrink_future, x2_value_of_waiting


def _known_out(pids: np.ndarray, day: pd.Timestamp, q: float) -> np.ndarray:
    d = int(day.value // 86_400_000_000_000)
    h = ((pids.astype(np.int64) * 2654435761) ^ (d * 40503)) % 1000
    return h < q * 1000


def eval_x2(picks: pd.DataFrame, samples: np.ndarray) -> dict:
    res = {"x2 le 1er jour du mois": [], "x2 sur le jour de plus forte EV (hindsight EV)": [],
           "x2 règle d'arrêt (DP)": [], "x2 jour au hasard (moyenne)": []}
    for _, g in picks.groupby([picks["date"].dt.year, picks["date"].dt.month]):
        if g["date"].iloc[0].month not in X2_MONTHS:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        res["x2 le 1er jour du mois"].append(g["actual"].iloc[0])
        res["x2 sur le jour de plus forte EV (hindsight EV)"].append(g.loc[g["ev"].idxmax(), "actual"])
        res["x2 jour au hasard (moyenne)"].append(g["actual"].mean())
        for i, r in g.iterrows():
            remaining = len(g) - 1 - i
            if remaining == 0 or r["ev"] >= x2_value_of_waiting([], remaining, samples):
                res["x2 règle d'arrêt (DP)"].append(r["actual"])
                break
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}


def run(hold=None, known_out=0.8, horizon=LOCK_GAP_DAYS, gap=LOCK_GAP_DAYS, max_days=None, top_n=12, shrink=0.5):
    from .data import load_all

    games = prepare_games(load_all())
    hold = hold or holdout_season(games)
    fb, cond, pp = build_tables(games)
    model = TTFLModel().fit(cond[cond["SEASON"] < hold], pp[pp["SEASON"] < hold])

    hg = games[games["SEASON"] == hold]
    tg_h = fb.tg[fb.tg["SEASON"] == hold]
    actual = {(int(p), d): v for p, d, v in zip(hg["PLAYER_ID"], hg["GAME_DATE"], hg["TTFL"])}
    appear = hg.groupby("GAME_DATE")["PLAYER_ID"].apply(set).to_dict()
    dates = sorted(hg["GAME_DATE"].unique())
    if max_days:
        dates = dates[:max_days]

    strategies = ["greedy_l10", "greedy_model", "planner"]
    locks = {s: {} for s in strategies}
    log = {s: [] for s in strategies}
    hindsight = []

    for n, t in enumerate(dates):
        t = pd.Timestamp(t)
        win = tg_h[(tg_h["GAME_DATE"] >= t) & (tg_h["GAME_DATE"] < t + pd.Timedelta(days=horizon))]
        roster = fb.roster(t, top_n=top_n)
        tgt = roster[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].merge(
            win[["TEAM_ID", "GAME_DATE", "OPP_TEAM_ID", "HOME"]], on="TEAM_ID")
        today = tgt["GAME_DATE"] == t
        absent = today & ~tgt["PLAYER_ID"].isin(appear.get(t, set()))
        known = absent & _known_out(tgt["PLAYER_ID"].to_numpy(), t, known_out)
        out = roster[roster["PLAYER_ID"].isin(tgt.loc[known, "PLAYER_ID"])]
        mates = out.groupby("TEAM_ID").agg(MATES_OUT_TTFL=("L10", "sum"), MATES_OUT_N=("L10", "size"))
        tgt = tgt[~known].copy()
        for c in ("MATES_OUT_TTFL", "MATES_OUT_N"):
            tgt[c] = np.where(tgt["GAME_DATE"] == t, tgt["TEAM_ID"].map(mates[c]), 0.0)
        if tgt.empty:
            continue
        rows = fb.rows_for(tgt, as_of=t, future_sched=win[["TEAM_ID", "GAME_DATE"]])
        rows = model.predict(rows)
        rows["DAY"] = (rows["GAME_DATE"] - t).dt.days
        d0 = rows[rows["DAY"] == 0]
        if d0.empty:
            continue
        hindsight.append(max(actual.get((int(p), t), 0.0) for p in d0["PLAYER_ID"]))

        for s in strategies:
            lk = locks[s]
            free = d0[~d0["PLAYER_ID"].map(lambda p: p in lk and (t - lk[p]).days < gap)]
            if free.empty:
                continue
            if s == "greedy_l10":
                pick = free.loc[free["L10"].idxmax()]
            elif s == "greedy_model":
                pick = free.loc[free["EV"].idxmax()]
            else:
                av = available_from(lk, t, gap)
                p = plan(shrink_future(rows[["DAY", "PLAYER_ID", "EV"]], shrink), av, gap)
                p0 = p[p["DAY"] == 0]
                pick = free[free["PLAYER_ID"] == p0["PLAYER_ID"].iloc[0]].iloc[0] if len(p0) else free.loc[free["EV"].idxmax()]
            lk[int(pick["PLAYER_ID"])] = t
            log[s].append({"date": t, "player": pick["PLAYER_NAME"], "ev": float(pick["EV"]),
                           "actual": actual.get((int(pick["PLAYER_ID"]), t), 0.0)})
        if (n + 1) % 20 == 0:
            print(f"  ... {n + 1}/{len(dates)} soirs simulés", flush=True)

    lines = [f"Backtest saison {hold} : {len(dates)} soirs, verrou {gap} j, horizon {horizon} j, absences connues {known_out:.0%}", ""]
    lines.append(f"{'Stratégie':<16s}{'Total':>9s}{'Moy/soir':>10s}{'EV prévue':>11s}{'Zéros':>7s}{'Joueurs distincts':>20s}")
    base_total = None
    for s in strategies:
        d = pd.DataFrame(log[s])
        if d.empty:
            continue
        tot = d["actual"].sum()
        base_total = base_total or tot
        lines.append(f"{s:<16s}{tot:>9.0f}{d['actual'].mean():>10.2f}{d['ev'].mean():>11.2f}{(d['actual'] == 0).sum():>7d}{d['player'].nunique():>20d}"
                     f"   ({(tot / base_total - 1):+.1%} vs greedy_l10)")
    lines.append(f"{'(plafond hindsight, sans verrou)':<32s}{np.mean(hindsight):>6.2f} /soir")

    plan_log = pd.DataFrame(log["planner"])
    if not plan_log.empty:
        samples = plan_log["ev"].to_numpy()
        lines += ["", "Bonus x2 (gain moyen par mois, en points ajoutés, pick du planner) :"]
        for k, v in eval_x2(plan_log, samples).items():
            lines.append(f"  {k:<52s}: {v:6.1f}")
        os.makedirs("models", exist_ok=True)
        pd.DataFrame({"ev": samples}).to_csv(os.path.join("models", "ttfl_ev_samples.csv"), index=False)
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join("models", "ttfl_backtest.txt"), "w") as f:
        f.write(report + "\n")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--known-out", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=LOCK_GAP_DAYS)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--shrink", type=float, default=0.5, help="lissage des EV futures (1 = aucun, 0 = moyenne du joueur)")
    a = ap.parse_args()
    run(a.holdout, a.known_out, a.horizon, LOCK_GAP_DAYS, a.max_days, shrink=a.shrink)
