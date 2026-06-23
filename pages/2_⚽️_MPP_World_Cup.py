import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from math import exp, factorial

sys.path.append(os.path.abspath('src'))
from mpp_oracle import prepare_live_inference_features

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")
st.title("⚽ Moteur d'Espérance de Gain (EV + Bonus Scores)")

# --- MATRICE DES BONUS MPP (Rareté des scores) ---
MPP_BONUS_EXACT = {
    (0,0): 30, (1,1): 15, (2,2): 40, (3,3): 80,
    (1,0): 20, (2,0): 25, (2,1): 20, (3,0): 35, (3,1): 35, (3,2): 45, (4,0): 50, (4,1): 50, (4,2): 60,
    (0,1): 20, (0,2): 25, (1,2): 20, (0,3): 35, (1,3): 35, (2,3): 45, (0,4): 50, (1,4): 50, (2,4): 60
}

def poisson_prob(l, k):
    return (l**k * exp(-l)) / factorial(k)

def calculate_exact_score_ev(goal_diff, prob_1x2):
    """
    Calcule l'Espérance du Bonus de Score.
    goal_diff = T1_goals - T2_goals
    On part sur une moyenne de 2.5 buts/match dans le tournoi pour distribuer.
    """
    l1 = max(0.1, (2.5 + goal_diff) / 2)
    l2 = max(0.1, (2.5 - goal_diff) / 2)
    
    score_probs = {}
    expected_bonus = {'T1': 0.0, 'NUL': 0.0, 'T2': 0.0}
    best_score_str = ""
    max_score_ev = -1
    
    for i in range(5):
        for j in range(5):
            prob = poisson_prob(l1, i) * poisson_prob(l2, j)
            score_probs[(i,j)] = prob
            bonus = MPP_BONUS_EXACT.get((i,j), 60) # Par défaut 60 pour les scores improbables (5-0)
            
            ev = prob * bonus
            if i > j: expected_bonus['T1'] += ev
            elif i == j: expected_bonus['NUL'] += ev
            else: expected_bonus['T2'] += ev
            
            # On cherche le score exact qui a la plus grosse EV brute pour l'affichage
            if ev > max_score_ev:
                max_score_ev = ev
                best_score_str = f"{i} - {j}"
                
    return expected_bonus, best_score_str

@st.cache_resource
def load_oracle_environment():
    # CHARGEMENT DU MODÈLE CALIBRÉ (Finding A3/C3)
    clf = joblib.load('models/football_classifier_calibrated.pkl')
    reg = joblib.load('models/football_regressor.pkl')
    with open('models/football_features.txt', 'r') as f:
        features = [line.strip() for line in f.read().split(',') if line.strip()]
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    return clf, reg, features, df

@st.cache_data
def load_upcoming_matches():
    try:
        sched = pd.read_csv('data/wc_2026_results.csv', names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
        sched['DATE'] = pd.to_datetime(sched['DATE'])
        return sched[sched['HG'].isna() | sched['AG'].isna()].sort_values('DATE')
    except: return pd.DataFrame()

clf, reg, features_list, history_df = load_oracle_environment()
upcoming_matches = load_upcoming_matches()

if clf is None: st.stop()

st.sidebar.header("🎯 Ta Situation au Classement")
points_gap = st.sidebar.number_input("Écart de points", value=-560, step=10)

if upcoming_matches.empty:
    st.info("🗓️ Aucun match à venir détecté.")
    st.stop()

next_date = upcoming_matches['DATE'].iloc[0]
st.subheader(f"🗓️ Grille Automatique du {next_date.strftime('%d/%m/%Y')}")

for idx, match in upcoming_matches[upcoming_matches['DATE'] == next_date].iterrows():
    t1, t2 = match['HOME'], match['AWAY']
    
    with st.expander(f"🔮 ANALYSER : {t1} vs {t2}", expanded=True):
        c1, c2, c3 = st.columns(3)
        cote_t1 = c1.number_input(f"Cote {t1}", value=50, step=1, key=f"t1_{idx}")
        cote_nul = c2.number_input("Cote NUL", value=100, step=1, key=f"nul_{idx}")
        cote_t2 = c3.number_input(f"Cote {t2}", value=50, step=1, key=f"t2_{idx}")
        
        if st.button(f"Calculer l'EV Totale pour {t1} vs {t2}", key=f"btn_{idx}"):
            elo_dict = {t: history_df[history_df['TEAM_ID'] == t].sort_values('MATCH_DATE').iloc[-1]['ELO_POST'] if not history_df[history_df['TEAM_ID'] == t].empty else 1500 for t in [t1, t2]}
            
            X_pred = prepare_live_inference_features(t1, t2, str(next_date.date()), history_df, elo_dict)
            X_pred['T1_IS_HOST'] = 0
            X_pred = X_pred[features_list]

            # Inférence
            probs = clf.predict_proba(X_pred)[0]
            goal_diff = reg.predict(X_pred)[0]
            prob_dict = {'T1': probs[2], 'NUL': probs[1], 'T2': probs[0]}
            
            # ÉTAPE CLÉ : Distribution Poisson et calcul de l'EV Totale (Issue + Bonus Score)
            expected_bonus, recommended_score = calculate_exact_score_ev(goal_diff, prob_dict)
            
            ev_totale = {
                'T1': (prob_dict['T1'] * cote_t1) + expected_bonus['T1'],
                'NUL': (prob_dict['NUL'] * cote_nul) + expected_bonus['NUL'],
                'T2': (prob_dict['T2'] * cote_t2) + expected_bonus['T2']
            }
            
            # Décision
            if points_gap >= 30: # Leader
                best = max(prob_dict, key=prob_dict.get)
            else: # Chasseur (On enlève le plancher restrictif, on suit l'EV totale pure)
                best = max(ev_totale, key=ev_totale.get)

            winner_name = t1 if best == 'T1' else (t2 if best == 'T2' else 'MATCH NUL')

            st.success(f"**Cible EV MAX : Jouer {winner_name.upper()}** (Score mathématique recommandé : {recommended_score})")
            
            r1, r2, r3 = st.columns(3)
            def render(col, label, p, ev_issue, ev_bonus, ev_tot, is_best):
                bg, border = ("#e8f5e9", "#2e7b32") if is_best else ("#f9fafb", "#e5e7eb")
                col.markdown(f"""
                <div style="background:{bg}; padding:10px; border:2px solid {border}; border-radius:5px; text-align:center; color:#111827;">
                    <strong>{label}</strong><br>
                    <span style="font-size:26px; font-weight:bold;">{p:.1%}</span><br>
                    <span style="font-size:12px; color:#4b5563;">EV Issue : {ev_issue:.1f} | E[Bonus] : {ev_bonus:.1f}</span><br>
                    <span style="font-size:16px; font-weight:bold;">EV TOTALE : {ev_tot:.1f} pts</span>
                </div>
                """, unsafe_allow_html=True)
                
            render(r1, t1, prob_dict['T1'], prob_dict['T1']*cote_t1, expected_bonus['T1'], ev_totale['T1'], best == 'T1')
            render(r2, "NUL", prob_dict['NUL'], prob_dict['NUL']*cote_nul, expected_bonus['NUL'], ev_totale['NUL'], best == 'NUL')
            render(r3, t2, prob_dict['T2'], prob_dict['T2']*cote_t2, expected_bonus['T2'], ev_totale['T2'], best == 'T2')