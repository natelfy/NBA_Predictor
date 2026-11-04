"""Générateur de données synthétiques au format de src/ttfl/data.py (pour les tests, PAS pour l'entraînement réel)."""
import os

import numpy as np
import pandas as pd

from src.ttfl.core import ttfl_score
from src.ttfl.data import KEEP

SKILL = [20, 16, 13, 10, 8.5, 7, 6, 5, 4.5, 4]
MINS = [34, 32, 30, 27, 24, 21, 18, 15, 12, 10]


def make_games(seed=0, seasons=("2023-24", "2024-25", "2025-26"), n_teams=12, games_per_team=44):
    rng = np.random.default_rng(seed)
    team_ids = [1610612737 + i for i in range(n_teams)]
    abbr = {t: f"T{i:02d}" for i, t in enumerate(team_ids)}
    def_factor = {t: rng.uniform(0.9, 1.1) for t in team_ids}
    players = {}
    for t in team_ids:
        players[t] = [dict(pid=1000 * (i + 1) + t % 1000, skill=SKILL[i] * rng.uniform(0.85, 1.15), base_min=MINS[i], inj=False)
                      for i in range(10)]
    rows = []
    for si, season in enumerate(seasons):
        start = pd.Timestamp(f"{season[:4]}-10-22")
        played = {t: 0 for t in team_ids}
        day, gno = 0, 0
        while min(played.values()) < games_per_team:
            order = sorted(rng.permutation(team_ids), key=lambda t: played[t])  # les moins servis d'abord
            k = min(len(order) // 2, int(rng.integers(3, 6)))
            date = start + pd.Timedelta(days=day)
            for j in range(k):
                home, away = order[2 * j], order[2 * j + 1]
                gno += 1
                gid = f"00{season[2:4]}{gno:05d}"
                for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                    played[team] += 1
                    for pl in players[team]:
                        if pl["inj"]:
                            pl["inj"] = rng.random() > 0.2
                        else:
                            pl["inj"] = rng.random() < 0.02
                        if pl["inj"] or rng.random() < 0.02:
                            continue
                        mins = pl["base_min"] * rng.uniform(0.85, 1.15)
                        u = pl["skill"] / 20 * (min(def_factor[opp], 1.1)) * (1.03 if is_home else 0.97) * mins / 36
                        fga = rng.poisson(14 * u); fgm = rng.binomial(fga, 0.46)
                        f3a = rng.binomial(fga, 0.35); f3m = min(rng.binomial(f3a, 0.36), fgm)
                        fta = rng.poisson(5 * u); ftm = rng.binomial(fta, 0.78)
                        reb = rng.poisson(6 * u); oreb = rng.binomial(reb, 0.25)
                        rows.append(dict(
                            SEASON=season, PLAYER_ID=pl["pid"], PLAYER_NAME=f"P{pl['pid']}", TEAM_ID=team,
                            TEAM_ABBREVIATION=abbr[team], GAME_ID=gid, GAME_DATE=date.strftime("%Y-%m-%d"),
                            MATCHUP=f"{abbr[team]} vs. {abbr[opp]}" if is_home else f"{abbr[team]} @ {abbr[opp]}",
                            MIN=round(mins, 1), FGM=fgm, FGA=fga, FG3M=f3m, FG3A=f3a, FTM=ftm, FTA=fta, OREB=oreb,
                            REB=reb, AST=rng.poisson(4 * u), STL=rng.poisson(1.0 * u), BLK=rng.poisson(0.7 * u),
                            TOV=rng.poisson(2 * u), PTS=2 * fgm + f3m + ftm))
            day += 1
        for t in team_ids:  # dérive de forme entre saisons
            for pl in players[t]:
                pl["skill"] *= rng.uniform(0.9, 1.1)
    df = pd.DataFrame(rows)
    df["TTFL"] = ttfl_score(df)
    return df[KEEP]


def write_dir(path, **kw):
    os.makedirs(path, exist_ok=True)
    df = make_games(**kw)
    for s, g in df.groupby("SEASON"):
        g.to_csv(os.path.join(path, f"games_{s}.csv"), index=False)
    return df
