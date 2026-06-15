import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")

st.title("⚽ Moteur de Points - Coupe du Monde 2026")
st.markdown("Algorithme Terminal : xG Plafonnés, ELO Symétrique, **Pénalités Environnementales** et **Théorème de Dixon-Coles**.")

# ==========================================
# 1. DICTIONNAIRES (CALENDRIER ET GÉOGRAPHIE)
# ==========================================
# Mise à jour massive : Amériques, Afrique, Asie et Moyen-Orient
LATAM_AFRICA_TEAMS = {
    'Mexico', 'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Ecuador', 'Paraguay', 'Panama', 'Curaçao',
    'Senegal', 'Morocco', 'Algeria', 'Egypt', 'Ghana', 'Ivory Coast', 'South Africa', 'Tunisia', 'Cape Verde', 'DR Congo',
    'Japan', 'South Korea', 'Qatar', 'Saudi Arabia', 'Iran', 'Iraq', 'Uzbekistan', 'Jordan'
}

CLIMATE_ZONES = {
    "Climat Tempéré (Optimal)": 1.0, 
    "Chaleur/Humidité (Miami, Houston...)": 0.88, 
    "Haute Altitude (Mexico, Guadalajara...)": 0.80
}

CALENDRIER = {
    "Jeudi 11 juin 2026": [
        {"team_a": "Mexico", "team_b": "South Africa", "score": "2-0", "host": "Mexico", "climate": "Haute Altitude (Mexico, Guadalajara...)"},
        {"team_a": "South Korea", "team_b": "Czechia", "score": "2-1", "host": "", "climate": "Climat Tempéré (Optimal)"}
    ],
    "Vendredi 12 juin 2026": [
        {"team_a": "Canada", "team_b": "Bosnia and Herzegovina", "score": "1-1", "host": "Canada", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "United States", "team_b": "Paraguay", "score": "4-1", "host": "United States", "climate": "Chaleur/Humidité (Miami, Houston...)"}
    ],
    "Samedi 13 juin 2026": [
        {"team_a": "Qatar", "team_b": "Switzerland", "score": "1-1", "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Brazil", "team_b": "Morocco", "score": "1-1", "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "Haiti", "team_b": "Scotland", "score": "0-1", "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Australia", "team_b": "Turkey", "score": "2-0", "host": "", "climate": "Climat Tempéré (Optimal)"}
    ],
    "Dimanche 14 juin 2026": [
        {"team_a": "Germany", "team_b": "Curaçao", "score": None, "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "Netherlands", "team_b": "Japan", "score": None, "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "Ivory Coast", "team_b": "Ecuador", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Sweden", "team_b": "Tunisia", "score": None, "host": "", "climate": "Haute Altitude (Mexico, Guadalajara...)"}
    ],
    "Lundi 15 juin 2026": [
        {"team_a": "Spain", "team_b": "Cape Verde", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Belgium", "team_b": "Egypt", "score": None, "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "Saudi Arabia", "team_b": "Uruguay", "score": None, "host": "", "climate": "Haute Altitude (Mexico, Guadalajara...)"},
        {"team_a": "Iran", "team_b": "New Zealand", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"}
    ],
    "Mardi 16 juin 2026": [
        {"team_a": "France", "team_b": "Senegal", "score": None, "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "Iraq", "team_b": "Norway", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Argentina", "team_b": "Algeria", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Austria", "team_b": "Jordan", "score": None, "host": "", "climate": "Haute Altitude (Mexico, Guadalajara...)"}
    ],
    "Mercredi 17 juin 2026": [
        {"team_a": "Portugal", "team_b": "DR Congo", "score": None, "host": "", "climate": "Chaleur/Humidité (Miami, Houston...)"},
        {"team_a": "England", "team_b": "Croatia", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Ghana", "team_b": "Panama", "score": None, "host": "", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Uzbekistan", "team_b": "Colombia", "score": None, "host": "", "climate": "Haute Altitude (Mexico, Guadalajara...)"}
    ]
}

POPULAR_SCORES = {(1,0):1.0, (0,1):1.0, (1,1):1.0, (2,1):0.8, (1,2):0.8, (2,0):0.7, (0,2):0.7, (0,0):0.7}
ALPHA = 0.70
MOTIVATION_LEVELS = {"Équipe Type (100%)": 1.0, "Légère Rotation (85%)": 0.85, "Équipe B / Démotivé (60%)": 0.60}
RHO = 0.18  # Paramètre de dépendance Dixon-Coles

# ==========================================
# 2. LOGIQUE STATISTIQUE ET ENVIRONNEMENTALE
# ==========================================
def load_football_data():
    if not os.path.exists('data/processed_football_games.csv'):
        return None, None
    df = pd.read_csv('data/processed_football_games.csv')
    stats = df.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()
    
    power = None
    if os.path.exists('data/squad_power.csv'):
        power = pd.read_csv('data/squad_power.csv')
        
    return stats, power

def calculate_lambdas(stat_a, stat_b, power_df, team_a_name, team_b_name, host_nation, mot_a, mot_b, climate_zone):
    off_a = stat_a.get('AVG_GOALS_10', 1.5)
    off_b = stat_b.get('AVG_GOALS_10', 1.5)
    def_a = stat_a.get('AVG_OPP_GOALS_10', 1.0)
    def_b = stat_b.get('AVG_OPP_GOALS_10', 1.0)
    
    off_a *= (0.4 + 0.6 * mot_a)
    off_b *= (0.4 + 0.6 * mot_b)
    
    climate_factor_a = CLIMATE_ZONES[climate_zone] if team_a_name not in LATAM_AFRICA_TEAMS and climate_zone != "Climat Tempéré (Optimal)" else 1.0
    climate_factor_b = CLIMATE_ZONES[climate_zone] if team_b_name not in LATAM_AFRICA_TEAMS and climate_zone != "Climat Tempéré (Optimal)" else 1.0

    off_a *= climate_factor_a
    def_a *= (2.0 - climate_factor_a) 
    
    off_b *= climate_factor_b
    def_b *= (2.0 - climate_factor_b)

    xg_a_base = max(0.2, off_a * def_b)
    xg_b_base = max(0.2, off_b * def_a)
    
    elo_a = stat_a.get('ELO_POST', 1500)
    elo_b = stat_b.get('ELO_POST', 1500)
    
    if host_nation == team_a_name: elo_a += 100
    elif host_nation == team_b_name: elo_b += 100
        
    if power_df is not None:
        power_a = power_df[power_df['Team'] == team_a_name]['Squad_Power_Index'].values
        power_b = power_df[power_df['Team'] == team_b_name]['Squad_Power_Index'].values
        
        p_a = power_a[0] if len(power_a) > 0 else 50.0
        p_b = power_b[0] if len(power_b) > 0 else 50.0
        
        p_a *= (mot_a * climate_factor_a)
        p_b *= (mot_b * climate_factor_b)
        
        power_diff = p_a - p_b
        elo_a += power_diff * 4

    elo_diff = elo_a - elo_b
    
    shift_factor = elo_diff / 400.0  
    xg_a = xg_a_base * (1 + 0.6 * max(0, shift_factor)) / (1 + 0.6 * max(0, -shift_factor))
    xg_b = xg_b_base * (1 + 0.6 * max(0, -shift_factor)) / (1 + 0.6 * max(0, shift_factor))
        
    xg_a = min(max(xg_a, 0.1), 4.5)
    xg_b = min(max(xg_b, 0.1), 4.5)
        
    return xg_a, xg_b, elo_a, elo_b

stats_df, power_df = load_football_data()
if stats_df is None:
    st.error("❌ Base de données introuvable.")
    st.stop()

valid_teams = set(stats_df['TEAM_ID'].unique())

# ==========================================
# 3. INTERFACE UTILISATEUR MPP
# ==========================================
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    date_selected = st.selectbox("📅 Sélectionner la journée", list(CALENDRIER.keys()))
with col_header_2:
    mode = st.radio("🎯 Stratégie", [
        "Sécurité Absolue (Probabilité Terrain)", 
        "Équilibré (EV Max)", 
        "Challenger Amorti"
    ])

st.markdown("---")

matchs_du_jour = CALENDRIER[date_selected]
analyzed_matches = []

for idx, match in enumerate(matchs_du_jour):
    ta = match['team_a']
    tb = match['team_b']
    default_climate = match.get('climate', 'Climat Tempéré (Optimal)')
    
    if match.get('score'):
        with st.container():
            st.success(f"✅ **Terminé :** {ta} **{match['score']}** {tb}")
            st.markdown("<br>", unsafe_allow_html=True)
        continue 

    if ta not in valid_teams or tb not in valid_teams:
        st.error(f"⚠️ Équipe non trouvée dans la base de données : {ta} ou {tb}.")
        continue
        
    stat_a = stats_df[stats_df['TEAM_ID'] == ta].iloc[0]
    stat_b = stats_df[stats_df['TEAM_ID'] == tb].iloc[0]
    
    with st.expander(f"⚽ {ta} vs {tb}", expanded=True):
        
        col_env1, col_env2, col_env3 = st.columns([2, 1, 1])
        with col_env1:
            climate = st.selectbox("🌡️ Conditions Géographiques", list(CLIMATE_ZONES.keys()), index=list(CLIMATE_ZONES.keys()).index(default_climate), key=f"clim_{idx}")
        with col_env2:
            m_str_a = st.selectbox(f"Effectif {ta}", list(MOTIVATION_LEVELS.keys()), key=f"mot_a_{idx}")
        with col_env3:
            m_str_b = st.selectbox(f"Effectif {tb}", list(MOTIVATION_LEVELS.keys()), key=f"mot_b_{idx}")
            
        mot_a, mot_b = MOTIVATION_LEVELS[m_str_a], MOTIVATION_LEVELS[m_str_b]
        
        la, lb, elo_a, elo_b = calculate_lambdas(stat_a, stat_b, power_df, ta, tb, match.get('host', ''), mot_a, mot_b, climate)
        
        MAX_GOALS = 15
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        
        # Brique Dixon-Coles (Correction psychologique des matchs nuls)
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                base_prob = poisson.pmf(i, la) * poisson.pmf(j, lb)
                
                if i == 0 and j == 0: tau = max(0, 1 - la * lb * RHO)
                elif i == 1 and j == 0: tau = max(0, 1 + la * RHO)
                elif i == 0 and j == 1: tau = max(0, 1 + lb * RHO)
                elif i == 1 and j == 1: tau = max(0, 1 - RHO)
                else: tau = 1.0
                
                matrix[i][j] = base_prob * tau
                
        # Normalisation indispensable de la matrice
        matrix /= np.sum(matrix)
                
        prob_oracle_a = np.sum(np.tril(matrix, -1))
        prob_oracle_draw = np.trace(matrix)
        prob_oracle_b = np.sum(np.triu(matrix, 1))
        
        default_pts_a = int(35 / prob_oracle_a) if prob_oracle_a > 0 else 200
        default_pts_draw = int(35 / prob_oracle_draw) if prob_oracle_draw > 0 else 200
        default_pts_b = int(35 / prob_oracle_b) if prob_oracle_b > 0 else 200

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2.5])
        
        with c1:
            st.markdown("**Points MPP (Gains)**")
            pts_a = st.number_input(f"Pts {ta}", value=default_pts_a, key=f"ca_{idx}", step=1)
            pts_draw = st.number_input(f"Pts Nul", value=default_pts_draw, key=f"cd_{idx}", step=1)
            pts_b = st.number_input(f"Pts {tb}", value=default_pts_b, key=f"cb_{idx}", step=1)
            
        with c2:
            st.markdown("**Foule MPP (%)**")
            foule_a = st.slider(f"% sur {ta}", 0, 100, int(prob_oracle_a * 100), key=f"fa_{idx}")
            foule_draw = st.slider(f"% sur Nul", 0, 100, int(prob_oracle_draw * 100), key=f"fd_{idx}")
            foule_b = st.slider(f"% sur {tb}", 0, 100, int(prob_oracle_b * 100), key=f"fb_{idx}")

        prob_dict = {'a': prob_oracle_a, 'draw': prob_oracle_draw, 'b': prob_oracle_b}
        ev_dict = {k: prob_dict[k] * pts_dict for k, pts_dict in zip(['a', 'draw', 'b'], [pts_a, pts_draw, pts_b])}
        crowd_dict = {'a': max(foule_a/100, 0.10), 'draw': max(foule_draw/100, 0.10), 'b': max(foule_b/100, 0.10)}
        
        if mode == "Sécurité Absolue (Probabilité Terrain)":
            pick = max(prob_dict, key=prob_dict.get)
        elif mode == "Équilibré (EV Max)":
            pick = max(ev_dict, key=ev_dict.get)
        else:
            leverage_dict = {}
            for k in prob_dict:
                if prob_dict[k] < 0.15: leverage_dict[k] = -1 
                else: leverage_dict[k] = ev_dict[k] / (crowd_dict[k] ** 0.6)
            pick = max(leverage_dict, key=leverage_dict.get)
            
        def _ok(outcome, i, j):
            return (outcome=='a' and i>j) or (outcome=='draw' and i==j) or (outcome=='b' and i<j)
            
        best_score, max_p = (0,0), -1.0
        rare_score, max_r = (0,0), -1.0
        
        for i in range(7):
            for j in range(7):
                if _ok(pick, i, j):
                    p_exact = matrix[i][j]
                    if p_exact > max_p:
                        max_p, best_score = p_exact, (i, j)
                    adj = p_exact * (1 - ALPHA * POPULAR_SCORES.get((i, j), 0.2))
                    if adj > max_r:
                        max_r, rare_score = adj, (i, j)

        labels = {'a': ta, 'draw': 'Match Nul', 'b': tb}
        
        with c3:
            st.markdown("**Oracle Analytics**")
            st.metric("PRONO MPP À COCHER", labels[pick].upper())
            st.write(f"Proba Réelle : **{prob_dict[pick]:.1%}**")
            
            if climate != "Climat Tempéré (Optimal)":
                if ta not in LATAM_AFRICA_TEAMS: st.caption(f"⚠️ **{ta}** : Alerte Physique (-20%)")
                if tb not in LATAM_AFRICA_TEAMS: st.caption(f"⚠️ **{tb}** : Alerte Physique (-20%)")

        with c4:
            st.markdown("**Objectif Score Exact**")
            st.info(f"🏆 **Score Mathématique : {best_score[0]} - {best_score[1]}** (Proba: {max_p:.1%})")

        analyzed_matches.append({
            'match': f"{ta} vs {tb}",
            'pick': labels[pick],
            'leverage': (ev_dict[pick] / (crowd_dict[pick] ** 0.6)) if prob_dict[pick] >= 0.15 else 0
        })

if analyzed_matches:
    st.markdown("---")
    st.header("♟️ Radar du Bonus x2")
    best_x2 = max(analyzed_matches, key=lambda x: x['leverage'])
    st.success(f"🚨 **LEVIER MAXIMAL DETECTÉ** : Si tu dois utiliser ton bonus x2 aujourd'hui, coche **{best_x2['pick'].upper()}** sur le match **{best_x2['match']}**.")