"""Modèles TTFL.

- MU / Q20 / Q80 : score TTFL attendu SI le joueur joue (LightGBM, régression + quantiles).
- P_PLAY        : probabilité de jouer, apprise sur les joueurs de rotation (repos, B2B, sortie précoce...).
                  Si on connaît le statut de l'injury report, il remplace la proba apprise.
- EV = P_PLAY x MU : espérance du pick (un joueur absent = 0 point ET 30 jours de verrou perdus).

Usage : python -m src.ttfl.model            # évalue sur la dernière saison complète puis ré-entraîne sur tout
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from .core import STATUS_P_PLAY
from .features import COND_FEATURES, PP_FEATURES, FeatureBuilder, grid_targets, prepare_games

MODEL_PATH = os.path.join("models", "ttfl_model.pkl")
METRICS_PATH = os.path.join("models", "ttfl_metrics.txt")
RETURNING_P_PLAY = 0.45  # n'a pas joué le dernier match de son équipe et aucun statut connu


def _reg(**kw):
    return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=80,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                         random_state=7, verbose=-1, **kw)


class TTFLModel:
    def __init__(self):
        self.mean = self.q20 = self.q80 = self.clf = None

    def fit(self, cond: pd.DataFrame, pp: pd.DataFrame) -> "TTFLModel":
        cond = cond.dropna(subset=["L10"])
        X, y = cond[COND_FEATURES], cond["TTFL"]
        self.mean = _reg().fit(X, y)
        self.q20 = _reg(objective="quantile", alpha=0.2).fit(X, y)
        self.q80 = _reg(objective="quantile", alpha=0.8).fit(X, y)
        self.clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                  random_state=7, verbose=-1).fit(pp[PP_FEATURES], pp["PLAYED"])
        return self

    def predict(self, rows: pd.DataFrame, status: Optional[pd.Series] = None) -> pd.DataFrame:
        """rows : sortie de FeatureBuilder.rows_for. status : Series alignée (out/doubtful/questionable/probable)."""
        r = rows.copy()
        X = r[COND_FEATURES]
        r["MU"] = self.mean.predict(X)
        r["Q20"] = np.minimum(self.q20.predict(X), r["MU"])
        r["Q80"] = np.maximum(self.q80.predict(X), r["MU"])
        p = self.clf.predict_proba(r[PP_FEATURES])[:, 1]
        p = np.where(r["PLAYED_PREV"] == 1, p, RETURNING_P_PLAY)
        if status is not None:
            override = status.reindex(r.index).str.lower().map(STATUS_P_PLAY)
            p = np.where(override.notna(), override, p)
        r["P_PLAY"] = p
        r["EV"] = r["P_PLAY"] * r["MU"]
        return r

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "TTFLModel":
        return joblib.load(path)


def build_tables(games: pd.DataFrame, fb: Optional[FeatureBuilder] = None):
    fb = fb or FeatureBuilder(games)
    tgt = games[["PLAYER_ID", "TEAM_ID", "OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "HOME", "SEASON", "TTFL", "MIN"]]
    cond = fb.rows_for(fb.with_mates_out(tgt))
    pp = fb.rows_for(grid_targets(games, fb.post, fb.tg))
    return fb, cond, pp[pp["PLAYED_PREV"] == 1]


def holdout_season(games: pd.DataFrame) -> str:
    n = games.groupby("SEASON")["GAME_ID"].nunique()
    full = n[n >= 1100]
    return full.index[-1] if len(full) else n.index[-1]


def evaluate(cond: pd.DataFrame, pp: pd.DataFrame, hold: str) -> tuple:
    tr_c, te_c = cond[cond["SEASON"] < hold], cond[cond["SEASON"] == hold].dropna(subset=["L10"])
    tr_p, te_p = pp[pp["SEASON"] < hold], pp[pp["SEASON"] == hold]
    m = TTFLModel().fit(tr_c, tr_p)
    te = m.predict(te_c)
    te_pp = te_p.copy()
    p = m.clf.predict_proba(te_pp[PP_FEATURES])[:, 1]
    base = tr_p["PLAYED"].mean()
    ll = lambda y, q: float(-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))

    lines = [f"Saison de validation : {hold} (entraînement sur les saisons précédentes)",
             f"Lignes test (joueur-match) : {len(te)}", ""]
    y = te["TTFL"]
    for name, pred in [("Modèle", te["MU"]), ("Baseline L10", te["L10"]),
                       ("Baseline moyenne lissée saison", te["BLEND"])]:
        lines.append(f"MAE  {name:<34s}: {np.mean(np.abs(y - pred)):.3f}   RMSE: {np.sqrt(np.mean((y - pred) ** 2)):.3f}")
    inside = ((y >= te["Q20"]) & (y <= te["Q80"])).mean()
    lines.append(f"Couverture de l'intervalle [Q20,Q80] (cible 60%) : {inside:.1%}")

    # métrique qui compte : quel score réel obtient-on en prenant le "meilleur" joueur de chaque soir ?
    pool = te[te["MIN10"] >= 24].copy()
    lines += ["", "Moyenne du score réel du pick n°1 de chaque soir (joueurs ayant joué, sans verrou) :"]
    for name, col in [("Modèle (MU)", "MU"), ("Baseline L10", "L10"), ("Baseline lissée", "BLEND")]:
        top = pool.loc[pool.groupby("GAME_DATE")[col].idxmax()]
        lines.append(f"  {name:<22s}: {top['TTFL'].mean():.2f}")
    top = pool.loc[pool.groupby("GAME_DATE")["TTFL"].idxmax()]
    lines.append(f"  {'Plafond (hindsight)':<22s}: {top['TTFL'].mean():.2f}")

    lines += ["", f"Probabilité de jouer (rotation) : taux de base {base:.3f}",
              f"LogLoss modèle {ll(te_pp['PLAYED'], p):.4f}  vs constante {ll(te_pp['PLAYED'], np.full(len(te_pp), base)):.4f}"]
    return "\n".join(lines), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    args = ap.parse_args()
    from .data import load_all

    games = prepare_games(load_all())
    fb, cond, pp = build_tables(games)
    hold = args.holdout or holdout_season(games)
    report, _ = evaluate(cond, pp, hold)
    print(report)
    os.makedirs("models", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        f.write(report + "\n")
    final = TTFLModel().fit(cond, pp)
    final.save()
    print(f"\nModèle final (toutes saisons) sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
