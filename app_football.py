import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="MPP Oracle - Mode Théorie des Jeux", layout="wide")
st.title("♟️ MPP Oracle : Théorie des Jeux & Expected Value")
st.markdown("L'objectif n'est pas d'avoir juste. L'objectif est de maximiser les points par rapport à la masse.")

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    return df.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()

@st.cache_resource
def load_models():
    clf = joblib.load('models/football_classifier.pkl')
    with open('models/football_features.txt', 'r') as f:
        features = f.read().split(',')
    return clf, features

stats_df, clf, features = load_data(), *load_models()
teams = sorted(stats_df['TEAM_ID'].unique())

# --- SAISIE DU CONTEXTE TACTIQUE ET SOCIAL ---
st.subheader("1. Paramétrage du Match & Théorie des Jeux")

col1, col2, col3 = st.columns(3)
with col1:
    team_a = st.selectbox("Équipe Domicile (A)", teams, index=teams.index('France') if 'France' in teams else 0)
    odd_a = st.number_input(f"Cote MPP {team_a}", value=1.80, step=0.05)
    own_a = st.slider(f"% estimé des joueurs MPP sur {team_a}", 0, 100, 70) # L'Ownership

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    odd_draw = st.number_input("Cote Nul", value=3.40, step=0.05)
    own_draw = st.slider("% estimé sur le Nul", 0, 100, 10)

with col3:
    team_b = st.selectbox("Équipe Extérieur (B)", teams, index=teams.index('Australia') if 'Australia' in teams else 1)
    odd_b = st.number_input(f"Cote MPP {team_b}", value=4.50, step=0.05)
    own_b = st.slider(f"% estimé des joueurs MPP sur {team_b}", 0, 100, 20)

st.markdown("---")
st.subheader("2. Indice de Motivation (Piège des Poules)")
matchday = st.radio("Journée de la compétition", ["J1 / J2 / Élimination directe", "J3 (Dernier match de poule)"])

mot_a, mot_b = 1.0, 1.0
if matchday == "J3 (Dernier match de poule)":
    c1, c2 = st.columns(2)
    with c1:
        pts_a = st.selectbox(f"Points de {team_a} avant le match", [0, 1, 2, 3, 4, 6])
        if pts_a >= 6: mot_a = 0.80  # Qualifié, rotation d'effectif = Baisse de 20% de force
        elif pts_a == 0: mot_a = 0.85 # Éliminé, démobilisation
        else: mot_a = 1.10 # Sur-motivation pour la qualification
    with c2:
        pts_b = st.selectbox(f"Points de {team_b} avant le match", [0, 1, 2, 3, 4, 6])
        if pts_b >= 6: mot_b = 0.80
        elif pts_b == 0: mot_b = 0.85
        else: mot_b = 1.10

if st.button("🧠 Lancer l'Algorithme Théorie des Jeux", use_container_width=True):
    stat_a = stats_df[stats_df['TEAM_ID'] == team_a].iloc[0]
    stat_b = stats_df[stats_df['TEAM_ID'] == team_b].iloc[0]
    
    pred_data = {}
    for feat in features:
        if feat == 'T1_IS_HOST': pred_data[feat] = [0]
        else:
            base_feat = feat.replace('_DIFF', '')
            try: pred_data[feat] = [stat_a[base_feat] - stat_b[base_feat]]
            except: pred_data[feat] = [0]
                
    probs = clf.predict_proba(pd.DataFrame(pred_data)[features])[0]
    
    # --- MODULATEUR DE MOTIVATION ---
    base_prob_b, base_prob_draw, base_prob_a = probs[0], probs[1], probs[2]
    
    # Ajustement probabiliste selon la motivation
    adj_prob_a = base_prob_a * mot_a
    adj_prob_b = base_prob_b * mot_b
    # Normalisation pour que la somme reste à 100%
    total_adj = adj_prob_a + base_prob_draw + adj_prob_b
    prob_a = adj_prob_a / total_adj
    prob_b = adj_prob_b / total_adj
    prob_draw = base_prob_draw / total_adj
    
    # --- THÉORIE DES JEUX (LEVERAGE SCORE) ---
    def calculate_leverage(prob, odd, ownership_pct):
        ev = (prob * odd) - 1
        ownership_decimal = max(ownership_pct, 1) / 100.0 # Éviter la division par 0
        # Le Leverage est l'EV divisée par la proportion de gens qui vont le jouer
        # Un choix très joué tue l'effet de levier. Un pari ignoré le multiplie.
        leverage = ev / ownership_decimal
        return ev, leverage

    ev_a, lev_a = calculate_leverage(prob_a, odd_a, own_a)
    ev_draw, lev_draw = calculate_leverage(prob_draw, odd_draw, own_draw)
    ev_b, lev_b = calculate_leverage(prob_b, odd_b, own_b)

    # --- AFFICHAGE ---
    st.header("🎯 Analyse des Leviers (Contrarian Value)")
    
    r1, r2, r3 = st.columns(3)
    def display_metrics(col, title, prob, ev, lev, ownership):
        with col:
            st.markdown(f"### {title}")
            st.metric("Probabilité Réelle (Post-Motivation)", f"{prob:.1%}")
            
            ev_color = "🟢" if ev > 0 else "🔴"
            st.write(f"**Expected Value (EV):** {ev_color} {ev:.2f}")
            
            # Le score de création de différence
            lev_color = "🔥" if lev > 0.5 else ("⚠️" if lev > 0 else "❌")
            st.info(f"**Leverage Score:** {lev_color} {lev:.2f}")
            st.caption(f"Propriété MPP: {ownership}%")

    display_metrics(r1, team_a, prob_a, ev_a, lev_a, own_a)
    display_metrics(r2, "Match Nul", prob_draw, ev_draw, lev_draw, own_draw)
    display_metrics(r3, team_b, prob_b, ev_b, lev_b, own_b)

    st.markdown("---")
    st.subheader("Verdict de la Théorie des Jeux")
    
    scores = {team_a: lev_a, "Nul": lev_draw, team_b: lev_b}
    best_play = max(scores, key=scores.get)
    
    if scores[best_play] > 0:
        st.success(f"🚨 **LEVIER MAXIMAL : Jouer {best_play.upper()}** 🚨\n\nC'est la faille mathématique de ce match. La probabilité que ce scénario arrive, croisée avec le fait que peu de joueurs de ta ligue vont oser le cocher, crée un différentiel de points massif en ta faveur sur le long terme.")
    else:
        st.error("💀 **MATCH TOXIQUE (NO BET)**. Les cotes MPP sont trop faibles ET la foule est du bon côté. Mathématiquement, tu ne feras aucune différence sur ta ligue ici. Limite la casse en jouant l'équipe favorite, et cherche ton levier sur un autre match.")