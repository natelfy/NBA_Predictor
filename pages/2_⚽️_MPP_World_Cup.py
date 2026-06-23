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

def calculate_exact_score_ev(goal_diff, total_goals=2.5):
    """
    Espérance du bonus de score exact, PAR issue, + meilleur score exact PAR issue.
    goal_diff   = buts T1 - buts T2 (prédit par le régresseur).
    total_goals = total de buts attendu (figé pour l'instant ; cf. roadmap = le rendre dynamique).
    Retourne (expected_bonus, best_score) où best_score['T1'|'NUL'|'T2'] = (i, j).
    Le meilleur score est cherché DANS chaque issue -> garantit la cohérence score/issue.
    """
    l1 = max(0.1, (total_goals + goal_diff) / 2)
    l2 = max(0.1, (total_goals - goal_diff) / 2)

    expected_bonus = {'T1': 0.0, 'NUL': 0.0, 'T2': 0.0}
    best_score = {'T1': (1, 0), 'NUL': (1, 1), 'T2': (0, 1)}  # replis sûrs
    best_ev = {'T1': -1.0, 'NUL': -1.0, 'T2': -1.0}

    for i in range(6):
        for j in range(6):
            prob = poisson_prob(l1, i) * poisson_prob(l2, j)
            bonus = MPP_BONUS_EXACT.get((i, j), 60)  # défaut 60 pour les scores rares hors matrice
            ev = prob * bonus
            issue = 'T1' if i > j else ('NUL' if i == j else 'T2')
            expected_bonus[issue] += ev
            # meilleur score DANS l'issue (et non plus le meilleur score global)
            if ev > best_ev[issue]:
                best_ev[issue] = ev
                best_score[issue] = (i, j)

    return expected_bonus, best_score

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
            
            # ÉTAPE CLÉ : Distribution Poisson -> E[bonus] par issue + meilleur score PAR issue
            expected_bonus, best_score = calculate_exact_score_ev(goal_diff)

            cotes = {'T1': cote_t1, 'NUL': cote_nul, 'T2': cote_t2}
            ev_issue = {k: prob_dict[k] * cotes[k] for k in cotes}
            ev_totale = {k: ev_issue[k] + expected_bonus[k] for k in cotes}

            # Proba de la FOULE (marché MPP) dévigée, déduite des cotes -> mesure l'edge réel
            inv = {k: 1.0 / cotes[k] for k in cotes}
            s_inv = sum(inv.values())
            implied = {k: inv[k] / s_inv for k in cotes}
            edge = {k: prob_dict[k] - implied[k] for k in cotes}

            model_pick = max(prob_dict, key=prob_dict.get)   # ce que croit le modèle calibré
            value_pick = max(ev_totale, key=ev_totale.get)   # meilleure EV totale (issue + bonus)

            # --- Décision : se différencier UNIQUEMENT sur un edge réel vs la foule ---
            # (sinon on suit le modèle : pas de nul/outsider forcé sur chaque match)
            # Seuils provisoires -> le sélecteur définitif sera la simulation P(top-3).
            EDGE_MIN, PROB_MIN = 0.05, 0.20
            if points_gap >= 30:   # Leader : on sécurise -> issue la plus probable
                best = model_pick
            else:                  # Chasseur : valeur SEULEMENT si edge réel et proba non négligeable
                if value_pick == model_pick or (edge[value_pick] >= EDGE_MIN and prob_dict[value_pick] >= PROB_MIN):
                    best = value_pick
                else:
                    best = model_pick

            is_contrarian = (best == value_pick) and (value_pick != model_pick)
            si, sj = best_score[best]              # score COHÉRENT avec l'issue choisie
            recommended_score = f"{si} - {sj}"
            winner_name = t1 if best == 'T1' else (t2 if best == 'T2' else 'MATCH NUL')

            badge = " · 🎯 COUP CONTRARIAN" if is_contrarian else ""
            st.success(f"**Cible : Jouer {winner_name.upper()}** (Score recommandé : {recommended_score}){badge}")
            if is_contrarian:
                st.caption(f"✅ Différenciation justifiée : le modèle donne **{prob_dict[best]:.0%}** à cette issue "
                           f"vs **{implied[best]:.0%}** pour la foule (edge **+{edge[best]*100:.0f} pts de %**). "
                           f"C'est ce type de pari qui fait remonter au classement.")
            else:
                st.caption("ℹ️ Pas d'edge net vs la foule → on suit la prédiction du modèle (pas de prise de risque inutile).")
            
            r1, r2, r3 = st.columns(3)
            def render(col, label, key, is_best):
                p, imp, eg = prob_dict[key], implied[key], edge[key]
                bg, border = ("#e8f5e9", "#2e7b32") if is_best else ("#f9fafb", "#e5e7eb")
                eg_color = "#15803d" if eg >= 0 else "#b91c1c"
                col.markdown(f"""
                <div style="background:{bg}; padding:10px; border:2px solid {border}; border-radius:5px; text-align:center; color:#111827;">
                    <strong>{label}</strong><br>
                    <span style="font-size:26px; font-weight:bold;">{p:.1%}</span><br>
                    <span style="font-size:12px; color:#4b5563;">Foule : {imp:.0%} · <span style="color:{eg_color}; font-weight:bold;">edge {eg*100:+.0f}</span></span><br>
                    <span style="font-size:12px; color:#4b5563;">EV Issue : {ev_issue[key]:.1f} | E[Bonus] : {expected_bonus[key]:.1f}</span><br>
                    <span style="font-size:16px; font-weight:bold;">EV TOTALE : {ev_totale[key]:.1f} pts</span>
                </div>
                """, unsafe_allow_html=True)

            render(r1, t1, 'T1', best == 'T1')
            render(r2, "NUL", 'NUL', best == 'NUL')
            render(r3, t2, 'T2', best == 'T2')