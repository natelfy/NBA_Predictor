import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath('src'))
from mpp_oracle import prepare_live_inference_features

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")
st.title("⚽ Football Oracle (Auto-Live EV Engine)")

# =====================================================================
# 1. CHARGEMENT DE L'ENVIRONNEMENT ET DU CALENDRIER
# =====================================================================
@st.cache_resource
def load_oracle_environment():
    clf = joblib.load('models/football_classifier.pkl')
    reg = joblib.load('models/football_regressor.pkl')
    with open('models/football_features.txt', 'r') as f:
        features = [line.strip() for line in f.read().split(',') if line.strip()]
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    return clf, reg, features, df

@st.cache_data
def load_upcoming_matches():
    """Lit le calendrier officiel et isole les matchs non joués."""
    try:
        sched = pd.read_csv('data/wc_2026_results.csv', names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
        sched['DATE'] = pd.to_datetime(sched['DATE'])
        # Un match à venir est un match où les buts (HG/AG) sont vides/NaN
        upcoming = sched[sched['HG'].isna() | sched['AG'].isna()].sort_values('DATE')
        return upcoming
    except Exception as e:
        st.error(f"Erreur de lecture du calendrier : {e}")
        return pd.DataFrame()

clf, reg, features_list, history_df = load_oracle_environment()
upcoming_matches = load_upcoming_matches()

if clf is None:
    st.stop()

# =====================================================================
# 2. PANNEAU LATÉRAL : CONTEXTE STRATÉGIQUE
# =====================================================================
st.sidebar.header("🎯 Ta Situation au Classement")
points_gap = st.sidebar.number_input("Écart de points", value=-560, step=10)

if points_gap >= 30:
    st.sidebar.success("🛡️ MODE LEADER : Minimisation du risque.")
elif points_gap <= -40:
    st.sidebar.error("⚔️ MODE CHASSEUR : Recherche d'anomalies (EV Max).")
else:
    st.sidebar.warning("⚖️ MODE NEUTRE : Équilibre Valeur/Probabilité.")

# =====================================================================
# 3. GÉNÉRATION AUTOMATIQUE DE LA GRILLE DU JOUR
# =====================================================================
if upcoming_matches.empty:
    st.info("🗓️ Aucun match à venir détecté dans le calendrier officiel.")
    st.stop()

# On isole les matchs de la date la plus proche
next_date = upcoming_matches['DATE'].iloc[0]
daily_matches = upcoming_matches[upcoming_matches['DATE'] == next_date]

st.subheader(f"🗓️ Grille Automatique du {next_date.strftime('%d/%m/%Y')}")
st.markdown("Saisis les cotes Mon Petit Prono pour déclencher l'Oracle.")

for idx, match in daily_matches.iterrows():
    t1, t2 = match['HOME'], match['AWAY']
    
    with st.expander(f"🔮 ANALYSER : {t1} vs {t2}", expanded=True):
        col1, col2, col3 = st.columns(3)
        cote_t1 = col1.number_input(f"Cote {t1}", value=50, step=1, key=f"t1_{idx}")
        cote_nul = col2.number_input("Cote NUL", value=100, step=1, key=f"nul_{idx}")
        cote_t2 = col3.number_input(f"Cote {t2}", value=50, step=1, key=f"t2_{idx}")
        
        if st.button(f"Lancer l'algorithme pour {t1} vs {t2}", key=f"btn_{idx}"):
            # Reconstruction ELO
            elo_dict = {}
            for t in [t1, t2]:
                team_hist = history_df[history_df['TEAM_ID'] == t].sort_values('MATCH_DATE')
                elo_dict[t] = team_hist.iloc[-1]['ELO_POST'] if not team_hist.empty else 1500

            # Inférence pure
            X_pred = prepare_live_inference_features(t1, t2, str(next_date.date()), history_df, elo_dict)
            X_pred['T1_IS_HOST'] = 0
            X_pred = X_pred[features_list]

            probs = clf.predict_proba(X_pred)[0]
            goal_diff = reg.predict(X_pred)[0]
            
            prob_t1, prob_nul, prob_t2 = probs[2], probs[1], probs[0]
            ev_t1, ev_nul, ev_t2 = prob_t1 * cote_t1, prob_nul * cote_nul, prob_t2 * cote_t2
            
            # Théorie des jeux
            choices, ev_choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}, {'T1': ev_t1, 'NUL': ev_nul, 'T2': ev_t2}
            
            if points_gap >= 30:
                best_choice = max(choices, key=choices.get)
            elif points_gap <= -40:
                best_choice = max(ev_choices, key=ev_choices.get)
                if (best_choice == 'T1' and prob_t1 < 0.15) or (best_choice == 'T2' and prob_t2 < 0.15) or (best_choice == 'NUL' and prob_nul < 0.15):
                    best_choice = max(choices, key=choices.get)
            else:
                valid_choices = {k: v for k, v in ev_choices.items() if choices[k] >= 0.28}
                best_choice = max(valid_choices, key=valid_choices.get) if valid_choices else max(choices, key=choices.get)

            # Score
            abs_diff = abs(goal_diff)
            if best_choice == 'T1':
                score = "1 - 0 ou 2 - 1" if abs_diff < 1.0 else ("2 - 0" if abs_diff < 2.0 else "3 - 0")
                winner = t1
            elif best_choice == 'T2':
                score = "0 - 1 ou 1 - 2" if abs_diff < 1.0 else ("0 - 2" if abs_diff < 2.0 else "0 - 3")
                winner = t2
            else:
                score = "0 - 0" if abs_diff < 0.5 else "1 - 1"
                winner = "MATCH NUL"

            # Rendu visuel
            st.success(f"**Cible : Jouer {winner.upper()}** (Score idéal : {score})")
            
            r1, r2, r3 = st.columns(3)
            def render_stat(col, label, p, ev, is_best):
                bg = "#e8f5e9" if is_best else "#f9fafb"
                border = "#2e7b32" if is_best else "#e5e7eb"
                col.markdown(f"""
                <div style="background:{bg}; padding:10px; border:2px solid {border}; border-radius:5px; text-align:center;">
                    <strong>{label}</strong><br>
                    <span style="font-size:24px;">{p:.1%}</span><br>
                    EV : {ev:.1f} pts
                </div>
                """, unsafe_allow_html=True)
                
            render_stat(r1, t1, prob_t1, ev_t1, best_choice == 'T1')
            render_stat(r2, "NUL", prob_nul, ev_nul, best_choice == 'NUL')
            render_stat(r3, t2, prob_t2, ev_t2, best_choice == 'T2')