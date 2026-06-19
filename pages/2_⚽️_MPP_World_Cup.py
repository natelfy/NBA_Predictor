import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")

st.title("⚽ Moteur de Points - Coupe du Monde 2026")
st.markdown("Algorithme Terminal : xG Plafonnés, Climat, Dixon-Coles, Tactique et Fracture d'Effectif.")

# ==========================================
# 1. DICTIONNAIRES ET CALENDRIER COMPLET
# ==========================================
LATAM_AFRICA_TEAMS = {
    'Mexico', 'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Ecuador', 'Paraguay', 'Panama', 'Curaçao',
    'Senegal', 'Morocco', 'Algeria', 'Egypt', 'Ghana', 'Ivory Coast', 'South Africa', 'Tunisia', 'Cape Verde', 'DR Congo',
    'Japan', 'South Korea', 'Qatar', 'Saudi Arabia', 'Iran', 'Iraq', 'Uzbekistan', 'Jordan'
}

CLIMATE_ZONES = {
    "Climat Tempéré (Optimal)": 1.0, 
    "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)": 0.88, 
    "Haute Altitude (Mexico, Guadalajara, Monterrey)": 0.80
}

TACTICAL_STYLES = ["Équilibré / Direct (Standard)", "Possession (Attaque Placée)", "Contre-Attaque (Bloc Bas)"]

SQUAD_INJURIES = {
    "Effectif Complet (100%)": (1.0, 1.0),
    "Star Offensive OUT": (0.85, 1.0),
    "Gardien Titulaire / Pilier Défensif OUT": (1.0, 1.25),
    "Hécatombe (3+ absents majeurs)": (0.80, 1.30)
}

CALENDRIER = {
    "Jeudi 11 juin 2026": [
        {"team_a": "Mexico", "team_b": "South Africa", "score": "2-0", "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"},
        {"team_a": "South Korea", "team_b": "Czech Republic", "score": "2-1", "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Vendredi 12 juin 2026": [
        {"team_a": "Canada", "team_b": "Bosnia and Herzegovina", "score": "1-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "United States", "team_b": "Paraguay", "score": "4-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"}
    ],
    "Samedi 13 juin 2026": [
        {"team_a": "Qatar", "team_b": "Switzerland", "score": "1-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Brazil", "team_b": "Morocco", "score": "1-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Haiti", "team_b": "Scotland", "score": "0-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Australia", "team_b": "Turkey", "score": "2-0", "climate": "Climat Tempéré (Optimal)"}
    ],
    "Dimanche 14 juin 2026": [
        {"team_a": "Germany", "team_b": "Curaçao", "score": "7-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Netherlands", "team_b": "Japan", "score": "2-2", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Ivory Coast", "team_b": "Ecuador", "score": "1-0", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Sweden", "team_b": "Tunisia", "score": "5-1", "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Lundi 15 juin 2026": [
        {"team_a": "Spain", "team_b": "Cape Verde", "score": "0-0", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Belgium", "team_b": "Egypt", "score": "1-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Saudi Arabia", "team_b": "Uruguay", "score": "1-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Iran", "team_b": "New Zealand", "score": "2-2", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"}
    ],
    "Mardi 16 juin 2026": [
        {"team_a": "France", "team_b": "Senegal", "score": "3-1", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Iraq", "team_b": "Norway", "score": "1-4", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Argentina", "team_b": "Algeria", "score": "3-0", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Austria", "team_b": "Jordan", "score": "3-1", "climate": "Climat Tempéré (Optimal)"}
    ],
    "Mercredi 17 juin 2026": [
        {"team_a": "Portugal", "team_b": "DR Congo", "score": "1-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "England", "team_b": "Croatia", "score": "4-2", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Ghana", "team_b": "Panama", "score": "1-0", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Uzbekistan", "team_b": "Colombia", "score": "1-3", "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Jeudi 18 juin 2026": [
        {"team_a": "Czech Republic", "team_b": "South Africa", "score": "1-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Switzerland", "team_b": "Bosnia and Herzegovina", "score": "4-1", "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Canada", "team_b": "Qatar", "score": "6-0", "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Mexico", "team_b": "South Korea", "score": "1-0", "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Vendredi 19 juin 2026": [
        {"team_a": "United States", "team_b": "Australia", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Scotland", "team_b": "Morocco", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Brazil", "team_b": "Haiti", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Turkey", "team_b": "Paraguay", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ],
    "Samedi 20 juin 2026": [
        {"team_a": "Netherlands", "team_b": "Sweden", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Germany", "team_b": "Ivory Coast", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Ecuador", "team_b": "Curaçao", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Tunisia", "team_b": "Japan", "score": None, "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Dimanche 21 juin 2026": [
        {"team_a": "Spain", "team_b": "Saudi Arabia", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Belgium", "team_b": "Iran", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Uruguay", "team_b": "Cape Verde", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "New Zealand", "team_b": "Egypt", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ],
    "Lundi 22 juin 2026": [
        {"team_a": "Argentina", "team_b": "Austria", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "France", "team_b": "Iraq", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Norway", "team_b": "Senegal", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Jordan", "team_b": "Algeria", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ],
    "Mardi 23 juin 2026": [
        {"team_a": "Portugal", "team_b": "Uzbekistan", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "England", "team_b": "Ghana", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Panama", "team_b": "Croatia", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Colombia", "team_b": "DR Congo", "score": None, "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Mercredi 24 juin 2026": [
        {"team_a": "Switzerland", "team_b": "Canada", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Bosnia and Herzegovina", "team_b": "Qatar", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Scotland", "team_b": "Brazil", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Morocco", "team_b": "Haiti", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Czech Republic", "team_b": "Mexico", "score": None, "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"},
        {"team_a": "South Africa", "team_b": "South Korea", "score": None, "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"}
    ],
    "Jeudi 25 juin 2026": [
        {"team_a": "Ecuador", "team_b": "Germany", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Curaçao", "team_b": "Ivory Coast", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Tunisia", "team_b": "Netherlands", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Japan", "team_b": "Sweden", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Turkey", "team_b": "United States", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Paraguay", "team_b": "Australia", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ],
    "Vendredi 26 juin 2026": [
        {"team_a": "Norway", "team_b": "France", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Senegal", "team_b": "Iraq", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Uruguay", "team_b": "Spain", "score": None, "climate": "Haute Altitude (Mexico, Guadalajara, Monterrey)"},
        {"team_a": "Cape Verde", "team_b": "Saudi Arabia", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "New Zealand", "team_b": "Belgium", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Egypt", "team_b": "Iran", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ],
    "Samedi 27 juin 2026": [
        {"team_a": "Panama", "team_b": "England", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Croatia", "team_b": "Ghana", "score": None, "climate": "Climat Tempéré (Optimal)"},
        {"team_a": "Colombia", "team_b": "Portugal", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "DR Congo", "team_b": "Uzbekistan", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Jordan", "team_b": "Argentina", "score": None, "climate": "Chaleur/Humidité (Miami, Houston, Dallas, LA, Atlanta...)"},
        {"team_a": "Algeria", "team_b": "Austria", "score": None, "climate": "Climat Tempéré (Optimal)"}
    ]
}

POPULAR_SCORES = {(1,0):1.0, (0,1):1.0, (1,1):1.0, (2,1):0.8, (1,2):0.8, (2,0):0.7, (0,2):0.7, (0,0):0.7}
ALPHA = 0.70
RHO = 0.18  

# ==========================================
# 2. LOGIQUE MATHÉMATIQUE
# ==========================================
def load_football_data():
    if not os.path.exists('data/processed_football_games.csv'): return None, None
    df = pd.read_csv('data/processed_football_games.csv')
    stats = df.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()
    power = pd.read_csv('data/squad_power.csv') if os.path.exists('data/squad_power.csv') else None
    return stats, power

def calculate_lambdas(stat_a, stat_b, power_df, team_a_name, team_b_name, climate_zone, style_a, style_b, inj_a, inj_b):
    off_a = stat_a.get('AVG_GOALS_10', 1.5)
    off_b = stat_b.get('AVG_GOALS_10', 1.5)
    def_a = stat_a.get('AVG_OPP_GOALS_10', 1.0)
    def_b = stat_b.get('AVG_OPP_GOALS_10', 1.0)
    
    climate_factor_a = CLIMATE_ZONES[climate_zone] if team_a_name not in LATAM_AFRICA_TEAMS and climate_zone != "Climat Tempéré (Optimal)" else 1.0
    climate_factor_b = CLIMATE_ZONES[climate_zone] if team_b_name not in LATAM_AFRICA_TEAMS and climate_zone != "Climat Tempéré (Optimal)" else 1.0

    off_a *= climate_factor_a
    def_a *= (2.0 - climate_factor_a) 
    off_b *= climate_factor_b
    def_b *= (2.0 - climate_factor_b)

    inj_mod_off_a, inj_mod_def_a = SQUAD_INJURIES[inj_a]
    inj_mod_off_b, inj_mod_def_b = SQUAD_INJURIES[inj_b]

    off_a *= inj_mod_off_a
    def_a *= inj_mod_def_a
    off_b *= inj_mod_off_b
    def_b *= inj_mod_def_b

    tac_mod_a, tac_mod_b = 1.0, 1.0
    if style_a == "Possession (Attaque Placée)" and style_b == "Contre-Attaque (Bloc Bas)": tac_mod_a, tac_mod_b = 0.85, 1.15
    elif style_b == "Possession (Attaque Placée)" and style_a == "Contre-Attaque (Bloc Bas)": tac_mod_a, tac_mod_b = 1.15, 0.85
    elif style_a == "Possession (Attaque Placée)" and style_b == "Équilibré / Direct (Standard)": tac_mod_a, tac_mod_b = 1.10, 0.90
    elif style_b == "Possession (Attaque Placée)" and style_a == "Équilibré / Direct (Standard)": tac_mod_a, tac_mod_b = 0.90, 1.10
    elif style_a == "Contre-Attaque (Bloc Bas)" and style_b == "Équilibré / Direct (Standard)": tac_mod_a, tac_mod_b = 0.90, 1.10
    elif style_b == "Contre-Attaque (Bloc Bas)" and style_a == "Équilibré / Direct (Standard)": tac_mod_a, tac_mod_b = 1.10, 0.90

    off_a *= tac_mod_a
    off_b *= tac_mod_b

    xg_a_base = max(0.2, off_a * def_b)
    xg_b_base = max(0.2, off_b * def_a)
    
    elo_a, elo_b = stat_a.get('ELO_POST', 1500), stat_b.get('ELO_POST', 1500)
        
    if power_df is not None:
        p_a = power_df[power_df['Team'] == team_a_name]['Squad_Power_Index'].values[0] if len(power_df[power_df['Team'] == team_a_name]) > 0 else 50.0
        p_b = power_df[power_df['Team'] == team_b_name]['Squad_Power_Index'].values[0] if len(power_df[power_df['Team'] == team_b_name]) > 0 else 50.0
        p_a *= (climate_factor_a * inj_mod_off_a)
        p_b *= (climate_factor_b * inj_mod_off_b)
        elo_a += (p_a - p_b) * 4

    shift_factor = (elo_a - elo_b) / 400.0  
    xg_a = xg_a_base * (1 + 0.6 * max(0, shift_factor)) / (1 + 0.6 * max(0, -shift_factor))
    xg_b = xg_b_base * (1 + 0.6 * max(0, -shift_factor)) / (1 + 0.6 * max(0, shift_factor))
        
    return min(max(xg_a, 0.1), 4.5), min(max(xg_b, 0.1), 4.5)

stats_df, power_df = load_football_data()
if stats_df is None: st.stop()
valid_teams = set(stats_df['TEAM_ID'].unique())

# ==========================================
# 3. INTERFACE UTILISATEUR MPP
# ==========================================
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1: date_selected = st.selectbox("📅 Sélectionner la journée", list(CALENDRIER.keys()), index=list(CALENDRIER.keys()).index("Mardi 16 juin 2026"))
with col_header_2: mode = st.radio("🎯 Stratégie", ["Sécurité Absolue", "Équilibré (EV Max)", "Challenger Amorti"])

st.markdown("---")

matchs_du_jour = CALENDRIER[date_selected]
analyzed_matches = []

for idx, match in enumerate(matchs_du_jour):
    ta, tb = match['team_a'], match['team_b']
    
    if match.get('score'): 
        st.success(f"✅ **Résultat Final :** {ta} **{match['score']}** {tb} (Climat: {match['climate']})")
        continue 
        
    if ta not in valid_teams or tb not in valid_teams: continue
        
    stat_a, stat_b = stats_df[stats_df['TEAM_ID'] == ta].iloc[0], stats_df[stats_df['TEAM_ID'] == tb].iloc[0]
    
    with st.expander(f"⚽ {ta} vs {tb} - {match.get('climate')}", expanded=True):
        
        col_env1, col_env2, col_env3 = st.columns([2, 1, 1])
        with col_env1: climate = st.selectbox("🌡️ Conditions Géographiques", list(CLIMATE_ZONES.keys()), index=list(CLIMATE_ZONES.keys()).index(match.get('climate', 'Climat Tempéré (Optimal)')), key=f"clim_{idx}")
        with col_env2: style_a = st.selectbox(f"Style {ta}", TACTICAL_STYLES, index=0, key=f"style_a_{idx}")
        with col_env3: style_b = st.selectbox(f"Style {tb}", TACTICAL_STYLES, index=0, key=f"style_b_{idx}")

        col_inj1, col_inj2 = st.columns(2)
        with col_inj1: inj_a = st.selectbox(f"Infirmerie {ta}", list(SQUAD_INJURIES.keys()), index=0, key=f"inj_a_{idx}")
        with col_inj2: inj_b = st.selectbox(f"Infirmerie {tb}", list(SQUAD_INJURIES.keys()), index=0, key=f"inj_b_{idx}")
            
        la, lb = calculate_lambdas(stat_a, stat_b, power_df, ta, tb, climate, style_a, style_b, inj_a, inj_b)
        
        MAX_GOALS = 15
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                base_prob = poisson.pmf(i, la) * poisson.pmf(j, lb)
                if i == 0 and j == 0: tau = max(0, 1 - la * lb * RHO)
                elif i == 1 and j == 0: tau = max(0, 1 + la * RHO)
                elif i == 0 and j == 1: tau = max(0, 1 + lb * RHO)
                elif i == 1 and j == 1: tau = max(0, 1 - RHO)
                else: tau = 1.0
                matrix[i][j] = base_prob * tau
                
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
        
        if mode == "Sécurité Absolue": pick = max(prob_dict, key=prob_dict.get)
        elif mode == "Équilibré (EV Max)": pick = max(ev_dict, key=ev_dict.get)
        else:
            leverage_dict = {k: (-1 if prob_dict[k] < 0.15 else ev_dict[k] / (crowd_dict[k] ** 0.6)) for k in prob_dict}
            pick = max(leverage_dict, key=leverage_dict.get)
            
        def _ok(outcome, i, j): return (outcome=='a' and i>j) or (outcome=='draw' and i==j) or (outcome=='b' and i<j)
            
        best_score, max_p = (0,0), -1.0
        rare_score, max_r = (0,0), -1.0
        
        for i in range(7):
            for j in range(7):
                if _ok(pick, i, j):
                    p_exact = matrix[i][j]
                    if p_exact > max_p: max_p, best_score = p_exact, (i, j)
                    adj = p_exact * (1 - ALPHA * POPULAR_SCORES.get((i, j), 0.2))
                    if adj > max_r: max_r, rare_score = adj, (i, j)

        labels = {'a': ta, 'draw': 'Match Nul', 'b': tb}
        
        with c3:
            st.markdown("**Oracle Analytics**")
            st.metric("PRONO MPP À COCHER", labels[pick].upper())
            st.write(f"Proba Réelle : **{prob_dict[pick]:.1%}**")
            
            if inj_a != "Effectif Complet (100%)" or inj_b != "Effectif Complet (100%)":
                st.caption(f"🚑 Fracture d'effectif appliquée")

        with c4:
            st.markdown("**Objectif Score Exact**")
            st.info(f"🏆 **Score Mathématique : {best_score[0]} - {best_score[1]}** (Proba: {max_p:.1%})")

        analyzed_matches.append({
            'match': f"{ta} vs {tb}", 
            'pick': labels[pick], 
            'prob': prob_dict[pick], 
            'ev': ev_dict[pick],
            'crowd': crowd_dict[pick]
        })

if analyzed_matches:
    st.markdown("---")
    st.header("♟️ Radar du Bonus x2 (Intelligence Hybride)")
    
    # Filtre de Sécurité Absolue : On exige au moins 55% de probabilité de victoire réelle
    safe_matches = [m for m in analyzed_matches if m['prob'] >= 0.55]
    
    if safe_matches:
        # Parmi ces matchs ultra-sécurisés, on cherche la plus haute Espérance de Gain (EV)
        best_x2 = max(safe_matches, key=lambda x: x['ev'])
        st.success(f"🚨 **CIBLE x2 VERROUILLÉE** : Coche **{best_x2['pick'].upper()}** sur le match **{best_x2['match']}**.")
        st.write(f"📊 *Justification : L'équipe franchit le mur de sécurité (Proba: {best_x2['prob']:.1%}) tout en garantissant la plus forte rentabilité du jour.*")
    else:
        st.warning("⚠️ **AUCUN x2 RECOMMANDÉ AUJOURD'HUI**. Tous les matchs sont soumis à une variance trop élevée (< 55% de chances). Garde ton bonus.")