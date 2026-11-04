"""Règles et constantes de la TTFL (source : https://fantasy.trashtalk.co/public/faq.php)."""
from __future__ import annotations

import pandas as pd

# Après un pick, le joueur est indisponible "pour les 30 jours suivants" (saison régulière).
# Lecture littérale : picks espacés d'au moins 31 jours calendaires. Si le deck TTFL montre qu'un
# joueur redevient dispo dès le 30e jour, passe cette constante à 30 (le seul endroit à changer).
LOCK_GAP_DAYS = 31

# Un bonus x2 par mois, de novembre à avril (perdu s'il n'est pas utilisé dans le mois).
X2_MONTHS = (11, 12, 1, 2, 3, 4)

# Probabilité de jouer selon le statut de l'injury report (vocabulaire du TTFL Advisor).
STATUS_P_PLAY = {
    "out": 0.0,            # Absent
    "doubtful": 0.07,      # Peu probable
    "questionable": 0.55,  # Incertain (50/50)
    "probable": 0.93,      # Probable
}


def ttfl_score(df: pd.DataFrame) -> pd.Series:
    """Score TTFL = PTS+REB+AST+STL+BLK+tirs réussis+3pts réussis+LF réussis
    - (balles perdues + tirs ratés + 3pts ratés + LF ratés).

    Les tirs à 3 points ratés comptent deux fois (tir raté ET 3pts raté), cf. exemple de la FAQ :
    28 pts, 10 reb, 6 ast, 2 stl, 1 blk, 11/20, 3/7 à 3pts, 3/4 LF, 4 TOV -> 46.
    """
    return (
        df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"]
        + df["FGM"] + df["FG3M"] + df["FTM"]
        - df["TOV"] - (df["FGA"] - df["FGM"]) - (df["FG3A"] - df["FG3M"]) - (df["FTA"] - df["FTM"])
    )


def season_of(date) -> str:
    """'2026-11-03' -> '2026-27'."""
    d = pd.Timestamp(date)
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def next_season(season: str) -> str:
    y = int(season[:4]) + 1
    return f"{y}-{(y + 1) % 100:02d}"


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"
