"""Collecte des box scores JOUEUR (nba_api) -> data/ttfl/games_<saison>.csv, avec score TTFL calculé.

Un fichier par saison : les saisons terminées ne changent plus, seul le fichier de la saison en cours
est réécrit (diffs git minuscules).

Usage :
    python -m src.ttfl.data                   # saisons manquantes + saison en cours
    python -m src.ttfl.data --seasons 2025-26 # forcer certaines saisons
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from .core import prev_season, season_of, ttfl_score

DATA_DIR = os.environ.get("TTFL_DATA_DIR", os.path.join("data", "ttfl"))
KEEP = [
    "SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
    "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "REB", "AST", "STL", "BLK",
    "TOV", "PTS", "TTFL",
]


def season_path(season: str) -> str:
    return os.path.join(DATA_DIR, f"games_{season}.csv")


def default_seasons(today=None, n_past: int = 4) -> list:
    cur = season_of(today or pd.Timestamp.today())
    out = [cur]
    for _ in range(n_past):
        out.append(prev_season(out[-1]))
    return sorted(out)


def fetch_season(season: str, retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="P",
                season_type_all_star="Regular Season", timeout=timeout,
            )
            df = log.get_data_frames()[0]
            break
        except Exception as e:  # réseau, blocage stats.nba.com, JSON vide...
            last_err = e
            print(f"  tentative {attempt}/{retries} échouée : {type(e).__name__}", file=sys.stderr)
            time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"Impossible de récupérer {season}: {last_err}")
    df["SEASON"] = season
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df[df["MIN"] > 0].copy()
    df["TTFL"] = ttfl_score(df)
    return df[KEEP].sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def load_all(data_dir: str = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    files = sorted(f for f in os.listdir(data_dir) if f.startswith("games_") and f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier games_*.csv dans {data_dir}. Lance : python -m src.ttfl.data")
    return pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)


def update(seasons=None) -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    cur = season_of(pd.Timestamp.today())
    wanted = seasons or default_seasons()
    failures = 0
    for s in wanted:
        path = season_path(s)
        if os.path.exists(path) and s != cur and not seasons:
            continue  # saison passée déjà en cache
        print(f"Saison {s} ...")
        try:
            df = fetch_season(s)
        except RuntimeError as e:
            print(f"  ECHEC : {e}", file=sys.stderr)
            failures += 1
            continue
        df.to_csv(path, index=False)
        print(f"  {len(df)} lignes joueur-match, dernier match {df['GAME_DATE'].max()}")
        time.sleep(1)
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None)
    args = ap.parse_args()
    sys.exit(1 if update(args.seasons) else 0)
