"""TTFL Advisor perso : EV par joueur, plan sous verrou 31 jours, conseil pour le x2.

Aide à la décision uniquement : tu poses tes picks toi-même sur fantasy.trashtalk.co (usage personnel, cf. CGU).
"""
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ttfl import schedule as sched_mod
from src.ttfl.advisor import candidates
from src.ttfl.core import LOCK_GAP_DAYS, X2_MONTHS
from src.ttfl.data import DATA_DIR, load_all
from src.ttfl.features import FeatureBuilder, prepare_games
from src.ttfl.model import MODEL_PATH, TTFLModel
from src.ttfl.planner import available_from, opportunity_costs, plan, shrink_future, x2_advice

st.set_page_config(page_title="TTFL Advisor", page_icon="🎯", layout="wide")
PICKS_PATH = os.path.join("data", "ttfl_picks.csv")
EV_SAMPLES = os.path.join("models", "ttfl_ev_samples.csv")
STATUS_LABELS = {"out": "Absent", "doubtful": "Peu probable", "questionable": "Incertain", "probable": "Probable"}
API_STATUS = {"out": "out", "doubtful": "doubtful", "questionable": "questionable", "probable": "probable",
              "day-to-day": "questionable"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


@st.cache_resource(show_spinner="Chargement des box scores et des variables...")
def load_engine(stamp):
    games = prepare_games(load_all())
    return games, FeatureBuilder(games)


@st.cache_resource
def load_model(stamp):
    return TTFLModel.load()


@st.cache_data(ttl=1800)
def api_injuries():
    try:
        from src.balldontlie_collector import BalldontlieCollector
        key = st.secrets.get("BALLDONTLIE_API_KEY", os.environ.get("BALLDONTLIE_API_KEY", ""))
        df = BalldontlieCollector(api_key=key).get_injuries()
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def stamp_of(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


st.title("🎯 TTFL Advisor")
st.caption("Score TTFL attendu par joueur, plan de picks sous verrou, conseil pour le bonus x2. "
           "Tu poses tes picks toi-même sur TTFL (pas d'automatisation).")

if not os.path.isdir(DATA_DIR) or not os.path.exists(MODEL_PATH) or not os.path.exists(sched_mod.SCHEDULE_PATH):
    st.error("Données ou modèle manquants. Lance dans l'ordre :")
    st.code("python -m src.ttfl.data\npython -m src.ttfl.schedule\npython -m src.ttfl.model", language="bash")
    st.stop()

games, fb = load_engine(max(stamp_of(os.path.join(DATA_DIR, f)) for f in os.listdir(DATA_DIR)))
model = load_model(stamp_of(MODEL_PATH))
team_games = sched_mod.load()
names = games.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"]
id_of = {norm(n): int(i) for i, n in names.items()}

with st.sidebar:
    today = pd.Timestamp(st.date_input("Soirée à préparer", value=pd.Timestamp.today().date()))
    horizon = st.slider("Horizon du plan (jours)", 7, 45, LOCK_GAP_DAYS)
    shrink = st.slider("Lissage des EV futures", 0.0, 1.0, 0.5, 0.1,
                       help="1 = EV brutes. Plus bas = on se méfie du « meilleur soir » estimé (biais du maximum).")
    st.caption(f"Dernier match en base : {games['GAME_DATE'].max().date()}")
    if (today - games["GAME_DATE"].max()).days > 3:
        st.warning("Données NBA anciennes : relance `python -m src.ttfl.data`.")

# ---- mes picks (verrous) -------------------------------------------------------------------
st.subheader("1. Mes picks récents (verrous)")
st.caption(f"Saisis tes picks des {LOCK_GAP_DAYS} derniers jours (et ceux déjà posés à l'avance). Le joueur redevient dispo "
           f"{LOCK_GAP_DAYS} jours après. Vérifie dans ton deck TTFL (colonne du nombre à côté du joueur).")
picks = pd.read_csv(PICKS_PATH, parse_dates=["date"]) if os.path.exists(PICKS_PATH) else pd.DataFrame(
    {"date": pd.Series(dtype="datetime64[ns]"), "player": pd.Series(dtype=str), "x2": pd.Series(dtype=bool)})
edited = st.data_editor(
    picks, num_rows="dynamic", use_container_width=True, key="picks",
    column_config={"date": st.column_config.DateColumn("Date"),
                   "player": st.column_config.SelectboxColumn("Joueur", options=sorted(names.unique())),
                   "x2": st.column_config.CheckboxColumn("x2 utilisé")})
if st.button("💾 Enregistrer mes picks"):
    os.makedirs("data", exist_ok=True)
    edited.dropna(subset=["date", "player"]).to_csv(PICKS_PATH, index=False)
    st.success("Enregistré.")
last_picks = {}
for _, r in edited.dropna(subset=["date", "player"]).iterrows():
    pid = id_of.get(norm(r["player"]))
    if pid is not None:
        last_picks[pid] = max(last_picks.get(pid, pd.Timestamp(r["date"])), pd.Timestamp(r["date"]))
avail = available_from(last_picks, today)

# ---- blessures -----------------------------------------------------------------------------
st.subheader("2. Blessures du soir")
inj = api_injuries()
default = {k: [] for k in STATUS_LABELS}
if len(inj) and {"player_name", "status"} <= set(inj.columns):
    for _, r in inj.iterrows():
        pid, st_ = id_of.get(norm(r["player_name"])), API_STATUS.get(str(r["status"]).lower().split()[0])
        if pid is not None and st_:
            default[st_].append(names[pid])
else:
    st.info("Injury report automatique indisponible : renseigne les statuts à la main (ceux du TTFL Advisor / NBA).")
c = st.columns(4)
sel = {k: c[i].multiselect(STATUS_LABELS[k], sorted(names.unique()), default=sorted(set(default[k])))
       for i, k in enumerate(STATUS_LABELS)}
statuses = {id_of[norm(n)]: k for k, v in sel.items() for n in v if norm(n) in id_of}

# ---- candidats / plan ----------------------------------------------------------------------
cands = candidates(fb, model, team_games, today, horizon, statuses)
if cands.empty:
    st.warning("Aucun match trouvé pour cette date dans le calendrier (relance `python -m src.ttfl.schedule`).")
    st.stop()
cands["PLAN_EV"] = shrink_future(cands[["DAY", "PLAYER_ID", "EV"]], shrink)["EV"]
planned = plan(cands[["DAY", "PLAYER_ID", "PLAN_EV"]].rename(columns={"PLAN_EV": "EV"}), avail)
cost = opportunity_costs(cands[["DAY", "PLAYER_ID", "PLAN_EV"]].rename(columns={"PLAN_EV": "EV"}), avail)

st.subheader(f"3. Pick du {today.date()}")
d0 = cands[cands["DAY"] == 0].copy()
if d0.empty:
    st.info("Pas de match ce jour-là.")
else:
    d0["DISPO"] = d0["PLAYER_ID"].map(lambda p: avail.get(p, 0) == 0)
    d0 = d0.merge(cost[["PLAYER_ID", "REGRET"]], on="PLAYER_ID", how="left")
    best_plan = planned[planned["DAY"] == 0]
    if len(best_plan):
        r = d0[d0["PLAYER_ID"] == best_plan["PLAYER_ID"].iloc[0]].iloc[0]
        st.success(f"**Pick conseillé (plan optimal) : {r['PLAYER_NAME']}** — EV {r['EV']:.1f} pts "
                   f"(si il joue : {r['MU']:.1f}, fourchette {r['Q20']:.0f}–{r['Q80']:.0f}, P(jouer) {r['P_PLAY']:.0%})")
    free = d0[d0["DISPO"]].sort_values("EV", ascending=False)
    if len(free):
        top = free.iloc[0]
        st.info(f"Meilleure EV brute du soir : **{top['PLAYER_NAME']}** ({top['EV']:.1f}). "
                f"« Manque à gagner » = points d'EV perdus sur tout le plan si on le prend ce soir plutôt que le plan optimal.")
    show = free.head(15)[["PLAYER_NAME", "EV", "MU", "Q20", "Q80", "P_PLAY", "L10", "MIN10", "B2B", "EXP_MARGIN", "REGRET"]]
    st.dataframe(show.rename(columns={
        "PLAYER_NAME": "Joueur", "EV": "EV", "MU": "Si il joue", "Q20": "Bas (20%)", "Q80": "Haut (80%)",
        "P_PLAY": "P(jouer)", "L10": "Moy. 10 m.", "MIN10": "Min. 10 m.", "B2B": "B2B",
        "EXP_MARGIN": "Écart prévu", "REGRET": "Manque à gagner"}).round(2), use_container_width=True, hide_index=True)
    st.caption("Joueurs verrouillés exclus : " + (", ".join(d0[~d0["DISPO"]]["PLAYER_NAME"]) or "aucun"))

    # ---- x2 ----------------------------------------------------------------------------------
    st.subheader("4. Bonus x2 du mois")
    if today.month in X2_MONTHS:
        used = edited[(edited["x2"] == True) & (pd.to_datetime(edited["date"]).dt.month == today.month)  # noqa: E712
                      & (pd.to_datetime(edited["date"]).dt.year == today.year)]
        if len(used):
            st.write("x2 déjà utilisé ce mois-ci.")
        elif len(best_plan):
            month_days = team_games[(team_games["GAME_DATE"].dt.month == today.month)
                                    & (team_games["GAME_DATE"].dt.year == today.year)]["GAME_DATE"].nunique()
            later = planned[(planned["DAY"] > 0) & ((today + pd.to_timedelta(planned["DAY"], unit="D")).dt.month == today.month)]
            known = list(later.sort_values("DAY")["EV"])
            days_left = team_games[(team_games["GAME_DATE"] > today) & (team_games["GAME_DATE"].dt.month == today.month)
                                   & (team_games["GAME_DATE"].dt.year == today.year)]["GAME_DATE"].nunique()
            n_unknown = max(0, days_left - len(known))
            samples = pd.read_csv(EV_SAMPLES)["ev"].to_numpy() if os.path.exists(EV_SAMPLES) else np.array(
                [best_plan["EV"].iloc[0]])
            adv = x2_advice(float(best_plan["EV"].iloc[0]), known, n_unknown, samples)
            (st.success if adv["use_today"] else st.info)(
                f"{'Utilise le x2 ce soir' if adv['use_today'] else 'Garde ton x2'} : EV du pick {adv['today_ev']:.1f} "
                f"vs valeur d'attente {adv['threshold']:.1f} ({days_left} soirs restants ce mois-ci, {month_days} au total). "
                "Active-le en dernier, juste avant la fermeture du deck (l'injury report bouge jusqu'au bout : "
                "un pick absent = 0 + x2 perdu).")
    else:
        st.write("Pas de x2 en dehors de novembre → avril.")

# ---- plan ----------------------------------------------------------------------------------
st.subheader(f"5. Plan des {horizon} prochains jours")
pl = planned.merge(cands[["DAY", "PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "MU", "P_PLAY", "EV"]].rename(columns={"EV": "EV_BRUTE"}),
                   on=["DAY", "PLAYER_ID"])
pl["Date"] = pl["GAME_DATE"].dt.date
st.dataframe(pl[["Date", "PLAYER_NAME", "EV_BRUTE", "MU", "P_PLAY"]].rename(columns={
    "PLAYER_NAME": "Joueur", "EV_BRUTE": "EV", "MU": "Si il joue", "P_PLAY": "P(jouer)"}).round(2),
    use_container_width=True, hide_index=True)
st.caption("Seul le pick du jour est à exécuter : le plan est recalculé chaque soir avec les blessures du moment. "
           "Les jours futurs ignorent les blessures à venir.")
