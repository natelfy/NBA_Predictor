import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath('src'))
from mpp_oracle import prepare_live_inference_features
import score_model as sm
import mpp_sim

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")
st.title("⚽ MPP Oracle — Moteur P(top-3) (EV + Bonus + Foule observée)")

OUTS = ['T1', 'NUL', 'T2']


# =====================================================================
# CHARGEMENT ENVIRONNEMENT (modèle calibré + ratings Poisson optionnels)
# =====================================================================
@st.cache_resource
def load_oracle_environment():
    clf = joblib.load('models/football_classifier_calibrated.pkl')
    reg = joblib.load('models/football_regressor.pkl')
    with open('models/football_features.txt', 'r') as f:
        features = [line.strip() for line in f.read().split(',') if line.strip()]
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    ratings = sm.load_ratings('data/team_ratings.json')   # None si fit_team_ratings.py pas lancé
    return clf, reg, features, df, ratings


@st.cache_data
def load_upcoming_matches():
    try:
        sched = pd.read_csv('data/wc_2026_results.csv', names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
        sched['DATE'] = pd.to_datetime(sched['DATE'])
        return sched[sched['HG'].isna() | sched['AG'].isna()].sort_values('DATE')
    except Exception as e:
        st.error(f"Erreur calendrier : {e}")
        return pd.DataFrame()


clf, reg, features_list, history_df, ratings = load_oracle_environment()
upcoming_matches = load_upcoming_matches()
if clf is None:
    st.stop()


# =====================================================================
# SIDEBAR : situation au classement (cible = combler l'écart au 3e)
# =====================================================================
st.sidebar.header("🎯 Ta situation")
gap_3e = st.sidebar.number_input("Écart au 3e place (points à combler)", value=505, step=10)
gap_1er = st.sidebar.number_input("Écart au 1er (contexte)", value=560, step=10)
competitor = st.sidebar.selectbox(
    "Modèle du concurrent (ligne du 3e)",
    options=['favori', 'valeur'],
    format_func=lambda x: "Suit la foule (réaliste)" if x == 'favori' else "Agressif (scénario prudent)")
trials = st.sidebar.select_slider("Précision (simulations)", options=[5000, 12000, 25000, 50000], value=12000)
st.sidebar.caption(f"team_ratings.json : {'✅ chargé (λ ajustées)' if ratings else '❌ absent (fallback goal_diff)'}")

if upcoming_matches.empty:
    st.info("🗓️ Aucun match à venir détecté.")
    st.stop()

next_date = upcoming_matches['DATE'].iloc[0]
daily = upcoming_matches[upcoming_matches['DATE'] == next_date]
st.subheader(f"🗓️ Grille du {next_date.strftime('%d/%m/%Y')}")
st.markdown("Saisis pour chaque match : **cotes MPP**, **% de la foule** (répartition des prons visible "
            "dans l'app) et, si tu les connais, les **bonus de score exact**. Puis lance la stratégie du jour.")


# =====================================================================
# HELPERS
# =====================================================================
def implied_from_cotes(cotes):
    inv = {k: 1.0 / max(cotes[k], 1e-9) for k in OUTS}
    s = sum(inv.values())
    return {k: inv[k] / s for k in OUTS}


def parse_bonus(s):
    """'0-0:30, 1-1:15' -> {(0,0):30.0,(1,1):15.0}"""
    d = {}
    for part in str(s).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            score, b = part.split(':')
            a, bb = score.strip().split('-')
            d[(int(a), int(bb))] = float(b)
        except Exception:
            pass
    return d


def model_probs_and_lambdas(t1, t2):
    sub1 = history_df[history_df['TEAM_ID'] == t1].sort_values('MATCH_DATE')
    sub2 = history_df[history_df['TEAM_ID'] == t2].sort_values('MATCH_DATE')
    elo_dict = {
        t1: (sub1.iloc[-1]['ELO_POST'] if not sub1.empty else 1500),
        t2: (sub2.iloc[-1]['ELO_POST'] if not sub2.empty else 1500),
    }
    X = prepare_live_inference_features(t1, t2, str(next_date.date()), history_df, elo_dict)
    X['T1_IS_HOST'] = 0
    X = X[features_list]
    probs = clf.predict_proba(X)[0]                      # [T2, NUL, T1]
    goal_diff = float(reg.predict(X)[0])
    p_model = {'T1': float(probs[2]), 'NUL': float(probs[1]), 'T2': float(probs[0])}
    if ratings is not None:
        lambdas = sm.lambdas_from_ratings(t1, t2, ratings, 0)
    else:
        lambdas = sm.lambdas_from_goal_diff(goal_diff)
    return p_model, lambdas


# =====================================================================
# SAISIE PAR MATCH (les valeurs persistent via les clés des widgets)
# =====================================================================
slate_inputs = []
for idx, match in daily.iterrows():
    t1, t2 = match['HOME'], match['AWAY']
    with st.expander(f"🔮 {t1} vs {t2}", expanded=True):
        c1, c2, c3 = st.columns(3)
        cote = {
            'T1': c1.number_input(f"Cote {t1}", value=50, step=1, key=f"cote_t1_{idx}"),
            'NUL': c2.number_input("Cote NUL", value=100, step=1, key=f"cote_nul_{idx}"),
            'T2': c3.number_input(f"Cote {t2}", value=50, step=1, key=f"cote_t2_{idx}"),
        }
        dflt = implied_from_cotes(cote)
        d1, d2, d3 = st.columns(3)
        crowd_pct = {
            'T1': d1.number_input(f"% foule {t1}", 0, 100, int(round(dflt['T1'] * 100)), key=f"cr_t1_{idx}"),
            'NUL': d2.number_input("% foule NUL", 0, 100, int(round(dflt['NUL'] * 100)), key=f"cr_nul_{idx}"),
            'T2': d3.number_input(f"% foule {t2}", 0, 100, int(round(dflt['T2'] * 100)), key=f"cr_t2_{idx}"),
        }
        bonus_str = st.text_input("Bonus de score connus (ex: 1-0:20, 0-0:30)", value="", key=f"bon_{idx}")
        slate_inputs.append({'t1': t1, 't2': t2, 'cote': cote, 'crowd_pct': crowd_pct, 'bonus_str': bonus_str})


# =====================================================================
# STRATÉGIE DU JOUR : modèle -> sim P(top-3) -> picks recommandés
# =====================================================================
if st.button("🎯 Calculer la stratégie du jour", type="primary"):
    slate = []
    rows = []
    for it in slate_inputs:
        t1, t2, cote = it['t1'], it['t2'], it['cote']
        p_model, lambdas = model_probs_and_lambdas(t1, t2)

        cp = it['crowd_pct']
        tot = max(cp['T1'] + cp['NUL'] + cp['T2'], 1)
        p_crowd = {k: cp[k] / tot for k in OUTS}            # normalisé à 1

        slate.append(mpp_sim.build_match(
            name=f"{t1} vs {t2}", p_model=p_model, p_crowd=p_crowd, cotes=cote,
            lambdas=lambdas, bonus_known=parse_bonus(it['bonus_str'])))

        edge = {k: p_model[k] - p_crowd[k] for k in OUTS}
        rows.append({
            'Match': f"{t1} vs {t2}",
            'Modèle': f"T1 {p_model['T1']:.0%} / N {p_model['NUL']:.0%} / T2 {p_model['T2']:.0%}",
            'Foule': f"T1 {p_crowd['T1']:.0%} / N {p_crowd['NUL']:.0%} / T2 {p_crowd['T2']:.0%}",
            'Meilleur edge': f"{max(edge, key=edge.get)} ({max(edge.values()) * 100:+.0f})",
        })
