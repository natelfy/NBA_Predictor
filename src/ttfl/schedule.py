"""Calendrier NBA (matchs à venir) -> data/ttfl/schedule.csv. Usage : python -m src.ttfl.schedule"""
from __future__ import annotations

import os
import sys

import pandas as pd

from .core import season_of
from .data import DATA_DIR

SCHEDULE_PATH = os.path.join(DATA_DIR, "schedule.csv")


def fetch_schedule(season: str, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2

    df = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=timeout).get_data_frames()[0]
    df = df[df["gameId"].astype(str).str.startswith("002")]  # saison régulière
    d = pd.to_datetime(df["gameDate"], errors="coerce").dt.normalize()
    out = pd.DataFrame({"GAME_ID": df["gameId"].astype(str), "GAME_DATE": d,
                        "HOME_TEAM_ID": df["homeTeam_teamId"], "AWAY_TEAM_ID": df["awayTeam_teamId"]})
    return out.dropna().astype({"HOME_TEAM_ID": int, "AWAY_TEAM_ID": int}).sort_values("GAME_DATE")


def to_team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (équipe, match) : TEAM_ID, OPP_TEAM_ID, HOME, GAME_DATE, GAME_ID."""
    s = sched.copy()
    s["GAME_DATE"] = pd.to_datetime(s["GAME_DATE"])
    h = s.rename(columns={"HOME_TEAM_ID": "TEAM_ID", "AWAY_TEAM_ID": "OPP_TEAM_ID"}).assign(HOME=1)
    a = s.rename(columns={"AWAY_TEAM_ID": "TEAM_ID", "HOME_TEAM_ID": "OPP_TEAM_ID"}).assign(HOME=0)
    return pd.concat([h, a], ignore_index=True)[["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_TEAM_ID", "HOME"]]


def update() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        df = fetch_schedule(season_of(pd.Timestamp.today()))
    except Exception as e:
        print(f"Calendrier indisponible : {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    df.to_csv(SCHEDULE_PATH, index=False)
    print(f"{len(df)} matchs, {df['GAME_DATE'].min().date()} -> {df['GAME_DATE'].max().date()}")
    return 0


def load() -> pd.DataFrame:
    return to_team_games(pd.read_csv(SCHEDULE_PATH, parse_dates=["GAME_DATE"]))


if __name__ == "__main__":
    sys.exit(update())
