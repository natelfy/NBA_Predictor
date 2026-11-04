"""Couche d'inférence : candidats des prochains jours avec EV, prêts pour le planificateur."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .core import LOCK_GAP_DAYS
from .features import FeatureBuilder
from .model import TTFLModel


def candidates(fb: FeatureBuilder, model: TTFLModel, team_games: pd.DataFrame, today, horizon: int = LOCK_GAP_DAYS,
               statuses: Optional[Dict[int, str]] = None, top_n: int = 12) -> pd.DataFrame:
    """team_games : sortie de schedule.to_team_games. statuses : {PLAYER_ID: out|doubtful|questionable|probable},
    appliqué au jour 0 uniquement (les statuts des jours suivants ne sont pas connus)."""
    today = pd.Timestamp(today).normalize()
    statuses = {int(k): v for k, v in (statuses or {}).items()}
    win = team_games[(team_games["GAME_DATE"] >= today) & (team_games["GAME_DATE"] < today + pd.Timedelta(days=horizon))]
    roster = fb.roster(today, top_n=top_n)
    tgt = roster[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].merge(
        win[["TEAM_ID", "GAME_DATE", "OPP_TEAM_ID", "HOME"]], on="TEAM_ID")
    if tgt.empty:
        return tgt
    out_ids = [p for p, s in statuses.items() if s == "out"]
    mates = roster[roster["PLAYER_ID"].isin(out_ids)].groupby("TEAM_ID").agg(
        MATES_OUT_TTFL=("L10", "sum"), MATES_OUT_N=("L10", "size"))
    for c in ("MATES_OUT_TTFL", "MATES_OUT_N"):
        tgt[c] = np.where(tgt["GAME_DATE"] == today, tgt["TEAM_ID"].map(mates[c]), 0.0)
    rows = fb.rows_for(tgt, as_of=today, future_sched=win[["TEAM_ID", "GAME_DATE"]])
    rows["DAY"] = (rows["GAME_DATE"] - today).dt.days
    status = pd.Series([statuses.get(int(p)) if d == 0 else None for p, d in zip(rows["PLAYER_ID"], rows["DAY"])],
                       index=rows.index, dtype="object")
    return model.predict(rows, status=status)
