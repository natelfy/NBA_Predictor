import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ttfl.core import ttfl_score
from src.ttfl.planner import plan, available_from, x2_value_of_waiting, x2_advice
from tests.synth_ttfl import make_games


class TestRules(unittest.TestCase):
    def test_faq_example_is_46(self):
        # 28 pts, 10 reb, 6 ast, 2 stl, 1 blk, 11/20 aux tirs dont 3/7 à 3pts, 3/4 LF, 4 TOV -> 46 (FAQ TTFL)
        df = pd.DataFrame([dict(PTS=28, REB=10, AST=6, STL=2, BLK=1, FGM=11, FGA=20, FG3M=3, FG3A=7,
                                FTM=3, FTA=4, TOV=4)])
        self.assertEqual(float(ttfl_score(df).iloc[0]), 46.0)

    def test_negative_score_possible(self):
        df = pd.DataFrame([dict(PTS=0, REB=0, AST=0, STL=0, BLK=0, FGM=0, FGA=6, FG3M=0, FG3A=4, FTM=0, FTA=2, TOV=3)])
        self.assertEqual(float(ttfl_score(df).iloc[0]), -(3 + 6 + 4 + 2))


class TestPlanner(unittest.TestCase):
    def test_plan_beats_greedy_and_respects_lock(self):
        # A (star) vaut 10 les deux jours, B vaut 9 puis 0. Glouton: A,B = 10. Optimal: B,A = 19.
        cand = pd.DataFrame([
            dict(DAY=0, PLAYER_ID=1, EV=10), dict(DAY=0, PLAYER_ID=2, EV=9),
            dict(DAY=1, PLAYER_ID=1, EV=10), dict(DAY=1, PLAYER_ID=2, EV=0)])
        p = plan(cand, {}, gap=31)
        self.assertEqual(p["EV"].sum(), 19)
        self.assertEqual(list(p.sort_values("DAY")["PLAYER_ID"]), [2, 1])

    def test_locked_player_excluded(self):
        cand = pd.DataFrame([dict(DAY=0, PLAYER_ID=1, EV=50), dict(DAY=0, PLAYER_ID=2, EV=5)])
        av = available_from({1: pd.Timestamp("2026-11-01")}, "2026-11-03", gap=31)
        self.assertEqual(av[1], 29)
        self.assertEqual(int(plan(cand, av, gap=31).iloc[0]["PLAYER_ID"]), 2)

    def test_same_player_twice_only_after_gap(self):
        cand = pd.DataFrame([dict(DAY=d, PLAYER_ID=1, EV=10.0) for d in (0, 5, 31, 40)]
                            + [dict(DAY=d, PLAYER_ID=2, EV=1.0) for d in (0, 5, 31, 40)])
        p = plan(cand, {}, gap=31)
        days1 = sorted(p[p["PLAYER_ID"] == 1]["DAY"])
        for a, b in zip(days1, days1[1:]):
            self.assertGreaterEqual(b - a, 31)

    def test_x2_threshold_grows_with_days_left(self):
        s = np.random.default_rng(0).normal(30, 8, 500)
        v = [x2_value_of_waiting([], k, s) for k in (0, 1, 5, 15)]
        self.assertEqual(v[0], 0.0)
        self.assertTrue(all(a < b for a, b in zip(v, v[1:])))
        self.assertTrue(x2_advice(60, [], 3, s)["use_today"])
        self.assertFalse(x2_advice(10, [55], 3, s)["use_today"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.ttfl.features import FeatureBuilder, prepare_games
        cls.games = prepare_games(make_games(seed=1))
        cls.fb = FeatureBuilder(cls.games)

    def test_no_future_leak(self):
        from src.ttfl.features import FeatureBuilder
        g = self.games
        row = g.iloc[len(g) // 2]
        tgt = lambda gm: gm[["PLAYER_ID", "TEAM_ID", "OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "HOME"]]
        base = self.fb.rows_for(tgt(g))
        # on truque le résultat de ce match ET de tous les matchs ultérieurs du joueur
        g2 = g.copy()
        mask = (g2["PLAYER_ID"] == row["PLAYER_ID"]) & (g2["GAME_DATE"] >= row["GAME_DATE"])
        g2.loc[mask, "TTFL"] += 1000
        alt = FeatureBuilder(g2).rows_for(tgt(g2))
        i = g.index[(g["PLAYER_ID"] == row["PLAYER_ID"]) & (g["GAME_ID"] == row["GAME_ID"])][0]
        cols = ["L5", "L10", "L20", "EWM", "SSN_AVG", "BLEND", "RATE10"]
        np.testing.assert_allclose(base.loc[i, cols].astype(float), alt.loc[i, cols].astype(float), equal_nan=True)
        later = g.index[(g["PLAYER_ID"] == row["PLAYER_ID"]) & (g["GAME_DATE"] > row["GAME_DATE"])][0]
        self.assertGreater(alt.loc[later, "L5"], base.loc[later, "L5"] + 100)

    def test_end_to_end(self):
        from src.ttfl.model import TTFLModel, build_tables, evaluate, holdout_season
        fb, cond, pp = build_tables(self.games, self.fb)
        hold = holdout_season(self.games)
        report, m = evaluate(cond, pp, hold)
        self.assertIn("Moyenne du score réel du pick n°1", report)
        rows = m.predict(cond.dropna(subset=["L10"]).head(50))
        self.assertTrue((rows["P_PLAY"].between(0, 1)).all())
        out = m.predict(cond.dropna(subset=["L10"]).head(5), status=pd.Series(["out"] * 5, index=cond.dropna(subset=["L10"]).head(5).index))
        self.assertTrue((out["EV"] == 0).all())

    def test_backtest_runs(self):
        from tests.synth_ttfl import write_dir
        with tempfile.TemporaryDirectory() as d:
            write_dir(d, seed=2)
            os.environ["TTFL_DATA_DIR"] = d
            import importlib, src.ttfl.data as data
            importlib.reload(data)
            import src.ttfl.backtest as bt
            importlib.reload(bt)
            cwd = os.getcwd()
            os.chdir(d)  # les rapports sont écrits dans ./models
            try:
                log = bt.run(max_days=12, horizon=10, gap=10)
            finally:
                os.chdir(cwd)
                os.environ.pop("TTFL_DATA_DIR", None)
        self.assertEqual(set(log), {"greedy_l10", "greedy_model", "planner"})
        for s, v in log.items():
            self.assertGreater(len(v), 5)
            names = [x["player"] for x in v]
            self.assertEqual(len(names), len(set(names)))  # verrou : jamais deux fois le même joueur sur 10 j


if __name__ == "__main__":
    unittest.main()


class TestAdvisor(unittest.TestCase):
    def test_candidates_from_future_schedule(self):
        from src.ttfl.advisor import candidates
        from src.ttfl.features import FeatureBuilder, prepare_games
        from src.ttfl.model import TTFLModel, build_tables
        from src.ttfl.planner import plan
        from src.ttfl.schedule import to_team_games
        games = prepare_games(make_games(seed=3))
        cut = games["GAME_DATE"].sort_values().iloc[int(len(games) * 0.9)]
        past, future = games[games["GAME_DATE"] < cut], games[games["GAME_DATE"] >= cut]
        fb = FeatureBuilder(past)
        _, cond, pp = build_tables(past, fb)
        model = TTFLModel().fit(cond, pp)
        g = future.drop_duplicates("GAME_ID")
        home = g[g["HOME"] == 1]
        sched = pd.DataFrame({"GAME_ID": home["GAME_ID"], "GAME_DATE": home["GAME_DATE"],
                              "HOME_TEAM_ID": home["TEAM_ID"], "AWAY_TEAM_ID": home["OPP_TEAM_ID"]})
        tgames = to_team_games(sched)
        star = int(past.sort_values("GAME_DATE").groupby("PLAYER_ID")["L10" if "L10" in past else "TTFL"].last().idxmax())
        c0 = candidates(fb, model, tgames, cut, horizon=10)
        self.assertFalse(c0.empty)
        self.assertTrue(c0["EV"].notna().all())
        out = candidates(fb, model, tgames, cut, horizon=10, statuses={star: "out"})
        d0 = out[(out["DAY"] == 0) & (out["PLAYER_ID"] == star)]
        self.assertTrue((d0["EV"] == 0).all())
        p = plan(c0[["DAY", "PLAYER_ID", "EV"]], {}, gap=10)
        self.assertEqual(p["DAY"].nunique(), c0["DAY"].nunique())
