"""Construction des variables, sans fuite du futur.

Principe : on calcule pour chaque ligne réelle un "état après match" (forme, minutes, rythme d'équipe...).
L'état *avant* un match est le dernier état strictement antérieur (merge_asof, allow_exact_matches=False).
Le même code sert donc à l'entraînement (matchs passés), au backtest (as_of) et à l'inférence (matchs à venir).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import next_season, season_of, ttfl_score

ROT_MIN = 18.0    # minutes (moy. 10 derniers matchs) à partir desquelles un joueur compte comme "rotation"
POOL_MIN = 12.0   # idem pour entrer dans le pool de candidats
DEFAULT_PRIOR = 9.0
PRIOR_K = 8.0     # poids (en matchs) de la saison précédente dans la moyenne lissée

COND_FEATURES = [
    "L5", "L10", "L20", "EWM", "STD10", "MIN5", "MIN10", "RATE10", "N_SSN", "SSN_AVG", "MIN_SSN",
    "BLEND", "PRI_TTFL", "LAST_MIN", "AVAIL20", "DAYS_SINCE", "HOME", "REST", "B2B", "OPP_REST",
    "OPP_B2B", "OPP_ALLOWED15", "OPP_PACE15", "PACE15", "FOR15", "EXP_MARGIN", "ABS_EXP_MARGIN",
    "MATES_OUT_TTFL", "MATES_OUT_N",
]
PP_FEATURES = [
    "AVAIL20", "MIN5", "MIN10", "LAST_MIN", "MIN_RATIO", "L10", "N_SSN", "REST", "B2B", "HOME", "DAYS_SINCE",
]


# --------------------------------------------------------------------------------------
# Préparation
# --------------------------------------------------------------------------------------
def prepare_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).astype("datetime64[ns]")
    for c in ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "REB", "AST", "STL", "BLK", "TOV", "PTS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "TTFL" not in df.columns:
        df["TTFL"] = ttfl_score(df)
    if "SEASON" not in df.columns:
        df["SEASON"] = df["GAME_DATE"].map(season_of)
    df = df[df["MIN"] > 0].dropna(subset=["TTFL"]).copy()
    df["HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    df["OPP_ABBR"] = df["MATCHUP"].str.split().str[-1]
    abbr2id = (df.sort_values("GAME_DATE").drop_duplicates("TEAM_ABBREVIATION", keep="last")
               .set_index("TEAM_ABBREVIATION")["TEAM_ID"])
    df["OPP_TEAM_ID"] = df["OPP_ABBR"].map(abbr2id)
    df = df.dropna(subset=["OPP_TEAM_ID"]).copy()
    df["OPP_TEAM_ID"] = df["OPP_TEAM_ID"].astype(int)
    df = df.drop_duplicates(["PLAYER_ID", "GAME_ID"])
    return df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def team_game_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["TEAM_ID", "GAME_ID"], as_index=False).agg(
        GAME_DATE=("GAME_DATE", "first"), SEASON=("SEASON", "first"), OPP_TEAM_ID=("OPP_TEAM_ID", "first"),
        HOME=("HOME", "first"), PTS=("PTS", "sum"), TTFL=("TTFL", "sum"), FGA=("FGA", "sum"),
        FTA=("FTA", "sum"), OREB=("OREB", "sum"), TOV=("TOV", "sum"),
    )
    agg["POSS"] = agg["FGA"] + 0.44 * agg["FTA"] - agg["OREB"] + agg["TOV"]
    opp = agg[["GAME_ID", "TEAM_ID", "PTS", "TTFL", "POSS"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "TTFL": "OPP_TTFL", "POSS": "OPP_POSS"})
    tg = agg.merge(opp, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    tg["MARGIN"] = tg["PTS"] - tg["OPP_PTS"]
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    g = tg.groupby("TEAM_ID")
    tg["TG_IDX"] = tg.groupby(["TEAM_ID", "SEASON"]).cumcount() + 1
    for col, name in [("MARGIN", "MARGIN15"), ("POSS", "PACE15"), ("OPP_TTFL", "ALLOWED15"), ("TTFL", "FOR15")]:
        tg[name] = g[col].transform(lambda s: s.rolling(15, min_periods=3).mean())
    nxt = g[["GAME_ID", "GAME_DATE", "OPP_TEAM_ID", "HOME", "SEASON"]].shift(-1)
    same = nxt["SEASON"] == tg["SEASON"]
    tg["NEXT_GAME_ID"] = nxt["GAME_ID"].where(same)
    tg["NEXT_DATE"] = nxt["GAME_DATE"].where(same)
    tg["NEXT_OPP"] = nxt["OPP_TEAM_ID"].where(same)
    tg["NEXT_HOME"] = nxt["HOME"].where(same)
    return tg


def player_post_state(df: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """État du joueur APRÈS chaque match (inclut ce match)."""
    p = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON", "MIN", "TTFL"]].copy()
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g = p.groupby("PLAYER_ID")

    def roll(col, w, fn="mean", min_periods=1):
        r = getattr(g[col].rolling(w, min_periods=min_periods), fn)()
        return r.reset_index(level=0, drop=True)

    for w in (5, 10, 20):
        p[f"L{w}"] = roll("TTFL", w)
    p["STD10"] = roll("TTFL", 10, "std", 4)
    p["MIN5"] = roll("MIN", 5)
    p["MIN10"] = roll("MIN", 10)
    p["RATE10"] = roll("TTFL", 10, "sum") / roll("MIN", 10, "sum")
    p["EWM"] = g["TTFL"].transform(lambda s: s.ewm(alpha=0.2, adjust=False).mean())

    gs = p.groupby(["PLAYER_ID", "SEASON"])
    p["N_SSN"] = gs.cumcount() + 1
    p["SSN_AVG"] = gs["TTFL"].transform(lambda s: s.expanding().mean())
    p["MIN_SSN"] = gs["MIN"].transform(lambda s: s.expanding().mean())
    p["LAST_MIN"] = p["MIN"]

    # a priori = moyenne de la saison précédente
    sm = p.groupby(["PLAYER_ID", "SEASON"], as_index=False).agg(PRI_TTFL=("TTFL", "mean"))
    sm["SEASON"] = sm["SEASON"].map(next_season)
    p = p.merge(sm, on=["PLAYER_ID", "SEASON"], how="left")
    prior = p["PRI_TTFL"].fillna(DEFAULT_PRIOR)
    p["BLEND"] = (p["N_SSN"] * p["SSN_AVG"] + PRIOR_K * prior) / (p["N_SSN"] + PRIOR_K)

    # disponibilité : part des 20 derniers matchs de SON équipe où il a joué
    p = p.merge(tg[["TEAM_ID", "GAME_ID", "TG_IDX"]], on=["TEAM_ID", "GAME_ID"], how="left")
    p["TG_IDX"] = p["TG_IDX"].fillna(1).astype(int)
    gid = p.groupby(["PLAYER_ID", "SEASON", "TEAM_ID"], sort=False).ngroup().to_numpy().astype(np.int64)
    idx = p["TG_IDX"].to_numpy().astype(np.int64)
    order = np.lexsort((idx, gid))
    key = gid[order] * 1000 + idx[order]
    left = np.searchsorted(key, np.maximum(key - 19, gid[order] * 1000), side="left")
    cnt = np.arange(len(key)) - left + 1
    avail = np.empty(len(p))
    avail[order] = cnt / np.minimum(20, idx[order])
    p["AVAIL20"] = np.clip(avail, 0, 1)

    p["LAST_DATE"] = p["GAME_DATE"]
    p["PL_TEAM"] = p["TEAM_ID"]
    p["NEXT_PTEAM"] = p.groupby("PLAYER_ID")["TEAM_ID"].shift(-1)
    return p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def mates_out_table(df: pd.DataFrame, post: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque match d'équipe : somme de la forme (L10) des joueurs de rotation ABSENTS ce soir-là."""
    rot = post[post["MIN10"] >= ROT_MIN][["PLAYER_ID", "TEAM_ID", "GAME_ID", "L10", "NEXT_PTEAM"]]
    rot = rot[rot["NEXT_PTEAM"].isna() | (rot["NEXT_PTEAM"] == rot["TEAM_ID"])]  # écarte les transferts
    rot = rot.merge(tg[["TEAM_ID", "GAME_ID", "NEXT_GAME_ID"]], on=["TEAM_ID", "GAME_ID"]).dropna(subset=["NEXT_GAME_ID"])
    appeared = df[["PLAYER_ID", "GAME_ID"]].assign(_a=1).rename(columns={"GAME_ID": "NEXT_GAME_ID"})
    rot = rot.merge(appeared, on=["PLAYER_ID", "NEXT_GAME_ID"], how="left")
    absent = rot[rot["_a"].isna()]
    out = absent.groupby(["TEAM_ID", "NEXT_GAME_ID"], as_index=False).agg(
        MATES_OUT_TTFL=("L10", "sum"), MATES_OUT_N=("L10", "size"))
    return out.rename(columns={"NEXT_GAME_ID": "GAME_ID"})


def grid_targets(df: pd.DataFrame, post: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Joueurs "de rotation" au match précédent x match d'équipe suivant, avec l'étiquette PLAYED.
    Sert à apprendre la probabilité de jouer (repos, back-to-back, sortie précoce...)."""
    r = post[post["MIN10"] >= POOL_MIN][["PLAYER_ID", "TEAM_ID", "GAME_ID", "NEXT_PTEAM"]]
    r = r[r["NEXT_PTEAM"].isna() | (r["NEXT_PTEAM"] == r["TEAM_ID"])]
    r = r.merge(tg[["TEAM_ID", "GAME_ID", "NEXT_GAME_ID", "NEXT_DATE", "NEXT_OPP", "NEXT_HOME", "SEASON"]],
                on=["TEAM_ID", "GAME_ID"]).dropna(subset=["NEXT_GAME_ID"])
    app = df[["PLAYER_ID", "GAME_ID"]].assign(PLAYED=1).rename(columns={"GAME_ID": "NEXT_GAME_ID"})
    r = r.merge(app, on=["PLAYER_ID", "NEXT_GAME_ID"], how="left")
    r["PLAYED"] = r["PLAYED"].fillna(0).astype(int)
    return r.rename(columns={"NEXT_GAME_ID": "GAME_ID", "NEXT_DATE": "GAME_DATE", "NEXT_OPP": "OPP_TEAM_ID",
                             "NEXT_HOME": "HOME"})[
        ["PLAYER_ID", "TEAM_ID", "OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "HOME", "SEASON", "PLAYED"]]


# --------------------------------------------------------------------------------------
# Constructeur
# --------------------------------------------------------------------------------------
def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str, cols: list, left_by: Optional[str] = None) -> pd.DataFrame:
    """Joint sur le dernier état STRICTEMENT antérieur à GAME_DATE."""
    left_by = left_by or by
    right = right[[by, "GAME_DATE"] + cols].sort_values("GAME_DATE", kind="mergesort")
    lft = left.sort_values("GAME_DATE", kind="mergesort")
    out = pd.merge_asof(lft, right, on="GAME_DATE", left_by=left_by, right_by=by,
                        allow_exact_matches=False, direction="backward")
    return out.sort_values("_i").reset_index(drop=True)


class FeatureBuilder:
    def __init__(self, games: pd.DataFrame):
        self.games = games
        self.tg = team_game_table(games)
        self.post = player_post_state(games, self.tg)
        self.mates_out = mates_out_table(games, self.post, self.tg)

    # -- tables filtrées "à la date as_of" (pas de fuite dans le backtest) --
    def _views(self, as_of):
        if as_of is None:
            return self.post, self.tg
        as_of = pd.Timestamp(as_of)
        return self.post[self.post["GAME_DATE"] < as_of], self.tg[self.tg["GAME_DATE"] < as_of]

    def rows_for(self, targets: pd.DataFrame, as_of=None, future_sched: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """targets : PLAYER_ID, TEAM_ID, OPP_TEAM_ID, GAME_DATE, HOME (+ MATES_OUT_TTFL/N facultatifs).
        future_sched : TEAM_ID, GAME_DATE des matchs à venir (pour repos / back-to-back)."""
        t = targets.copy().reset_index(drop=True)
        t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"]).astype("datetime64[ns]")
        t["_i"] = np.arange(len(t))
        for c in ("PLAYER_ID", "TEAM_ID", "OPP_TEAM_ID", "HOME"):
            t[c] = t[c].astype("int64")
        post, tg = self._views(as_of)

        pcols = ["L5", "L10", "L20", "EWM", "STD10", "MIN5", "MIN10", "RATE10", "N_SSN", "SSN_AVG", "MIN_SSN",
                 "BLEND", "PRI_TTFL", "LAST_MIN", "AVAIL20", "LAST_DATE", "PL_TEAM"]
        t = _asof(t, post, "PLAYER_ID", pcols)

        tcols = ["MARGIN15", "PACE15", "ALLOWED15", "FOR15"]
        tstate = tg[["TEAM_ID", "GAME_DATE"] + tcols].assign(TEAM_LAST_DATE=tg["GAME_DATE"])
        t = _asof(t, tstate, "TEAM_ID", tcols + ["TEAM_LAST_DATE"])
        ostate = tstate.rename(columns={"TEAM_ID": "OPP_TEAM_ID", **{c: f"OPP_{c}" for c in tcols}}).drop(
            columns=["TEAM_LAST_DATE"])
        t = _asof(t, ostate, "OPP_TEAM_ID", [f"OPP_{c}" for c in tcols])

        # repos / back-to-back d'après le calendrier (réel + à venir)
        parts = [tg[["TEAM_ID", "GAME_DATE"]], t[["TEAM_ID", "GAME_DATE"]],
                 t[["OPP_TEAM_ID", "GAME_DATE"]].rename(columns={"OPP_TEAM_ID": "TEAM_ID"})]
        if future_sched is not None and len(future_sched):
            fs = future_sched[["TEAM_ID", "GAME_DATE"]].copy()
            fs["GAME_DATE"] = pd.to_datetime(fs["GAME_DATE"]).astype("datetime64[ns]")
            parts.append(fs)
        sched = pd.concat(parts).drop_duplicates().sort_values(["TEAM_ID", "GAME_DATE"])
        sched["REST"] = sched.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(7).clip(1, 7)
        t = t.merge(sched, on=["TEAM_ID", "GAME_DATE"], how="left")
        o = sched.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "REST": "OPP_REST"})
        t = t.merge(o, on=["OPP_TEAM_ID", "GAME_DATE"], how="left")
        t["REST"] = t["REST"].fillna(7)
        t["OPP_REST"] = t["OPP_REST"].fillna(7)
        t["B2B"] = (t["REST"] == 1).astype(int)
        t["OPP_B2B"] = (t["OPP_REST"] == 1).astype(int)

        t["EXP_MARGIN"] = t["MARGIN15"] - t["OPP_MARGIN15"] + 2.5 * (2 * t["HOME"] - 1)
        t["ABS_EXP_MARGIN"] = t["EXP_MARGIN"].abs()
        t["OPP_ALLOWED15"] = t["OPP_ALLOWED15"]
        t["OPP_PACE15"] = t["OPP_PACE15"]
        t["DAYS_SINCE"] = (t["GAME_DATE"] - t["LAST_DATE"]).dt.days.clip(0, 21)
        t["MIN_RATIO"] = t["LAST_MIN"] / t["MIN10"].replace(0, np.nan)
        t["PLAYED_PREV"] = ((t["LAST_DATE"] == t["TEAM_LAST_DATE"]) & (t["PL_TEAM"] == t["TEAM_ID"])).astype(int)
        for c in ("MATES_OUT_TTFL", "MATES_OUT_N"):
            if c not in t.columns:
                t[c] = 0.0
            t[c] = t[c].fillna(0.0)
        return t

    def with_mates_out(self, targets: pd.DataFrame) -> pd.DataFrame:
        """Entraînement : rattache les absences réelles de rotation (par TEAM_ID, GAME_ID)."""
        return targets.merge(self.mates_out, on=["TEAM_ID", "GAME_ID"], how="left")

    # -- candidats pour un jour donné / une fenêtre --
    def roster(self, as_of, top_n: int = 12, max_idle_days: int = 14) -> pd.DataFrame:
        """Joueurs encore actifs à la date as_of (dernier état avant as_of), top_n par équipe."""
        post, _ = self._views(as_of)
        last = post.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
        idle = (pd.Timestamp(as_of) - last["LAST_DATE"]).dt.days
        # en début de saison, la dernière apparition date de la saison précédente : on garde les effectifs de l'an dernier
        new_season = last["SEASON"] != season_of(as_of)
        last = last[(idle <= max_idle_days) | (new_season & (idle <= 300))]
        last = last[last["MIN10"] >= POOL_MIN]
        last = last.sort_values("MIN10", ascending=False).groupby("TEAM_ID").head(top_n)
        return last[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "L10", "MIN10"]].reset_index(drop=True)
