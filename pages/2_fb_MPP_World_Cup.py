import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")

st.title("⚽ Moteur de Points - Coupe du Monde 2026")
st.markdown("Algorithme optimisé nativement pour les Points MPP. Maximisation de l'Expected Value.")

# ==========================================
# 1. DICTIONNAIRES ET CALENDRIER (Intacts)
# ==========================================
NATION_MAP = {
    "mexique": "Mexico", "afrique du sud": "South Africa", "république de corée": "South Korea", "tchéquie": "Czech Republic",
    "canada": "Canada", "bosnie-et-herzegovine": "Bosnia and Herzegovina", "bosnie-et-herzegovine ": "Bosnia and Herzegovina",
    "états-unis": "USA", "paraguay": "Paraguay", "qatar": "Qatar", "suisse": "Switzerland", "brésil": "Brazil", 
    "maroc": "Morocco", "haïti": "Haiti", "écosse": "Scotland", "australie": "Australia", "turquie": "Turkey", 
    "allemagne": "Germany", "curaçao": "Curaçao", "pays-bas": "Netherlands", "japon": "Japan", "côte d'ivoire": "Ivory Coast", 
    "équateur": "Ecuador", "suède": "Sweden", "tunisie": "Tunisia", "espagne": "Spain", "cap-vert": "Cape Verde", 
    "belgique": "Belgium", "égypte": "Egypt", "arabie saoudite": "Saudi Arabia", "uruguay": "Uruguay", "ri iran": "Iran", 
    "nouvelle-zélande": "New Zealand", "france": "France", "sénégal": "Senegal", "irak": "Iraq", "norvège": "Norway", 
    "argentine": "Argentina", "algérie": "Algeria", "autriche": "Austria", "jordanie": "Jordan", "portugal": "Portugal", 
    "rd congo": "DR Congo", "angleterre": "England", "croatie": "Croatia", "ghana": "Ghana", "panamá": "Panama", 
    "ouzbékistan": "Uzbekistan", "colombie": "Colombia", "czechia": "Czechia", "united states": "United States"
}

CALENDRIER = {
    "Jeudi 11 juin 2026": [
        {"team_a": "Mexico", "team_b": "South Africa", "score": "2-0"},
        {"team_a": "South Korea", "team_b": "Czechia", "score": "2-1"}
    ],
    "Vendredi 12 juin 2026": [
        {"team_a": "Canada", "team_b": "Bosnia and Herzegovina", "score": None, "host": "Canada"},
        {"team_a": "United States", "team_b": "Paraguay", "score": None, "host": "United States"}
    ],
    "Samedi 13 juin 2026": [
        {"team_a": "Qatar", "team_b": "Switzerland", "score": None},
        {"team_a": "Brazil", "team_b": "Morocco", "score": None},
        {"team_a": "Haiti", "team_b": "Scotland", "score": None},
        {"team_a": "Australia", "team_b": "Turkey", "score": None}
    ],
    "Dimanche 14 juin 2026": [
        {"team_a": "Germany", "team_b": "Curaçao", "score": None},
        {"team_a": "Netherlands", "team_b": "Japan", "score": None},
        {"team_a": "Ivory Coast", "team_b": "Ecuador", "score": None},
        {"team_a": "Sweden", "team_b": "Tunisia", "score": None}
    ],
    "Lundi 15 juin 2026": [
        {"team_a": "Spain", "team_b": "Cape Verde", "score": None},
        {"team_a": "Belgium", "team_b": "Egypt", "score": None},
        {"team_a": "Saudi Arabia", "team_b": "Uruguay", "score": None},
        {"team_a": "Iran", "team_b": "New Zealand", "score": None}
    ],
    "Mardi 16 juin 2026": [
        {"team_a": "France", "team_b": "Senegal", "score": None},
        {"team_a": "Iraq", "team_b": "Norway", "score": None},
        {"team_a": "Argentina", "team_b": "Algeria", "score": None},
        {"team_a": "Austria", "team_b": "Jordan", "score": None}
    ],
    "Mercredi 17 juin 2026": [
        {"team_a": "Portugal", "team_b": "DR Congo", "score": None},
        {"team_a": "England", "team_b": "Croatia", "score": None},
        {"team_a": "Ghana", "team_b": "Panama", "score": None},
        {"team_a": "Uzbekistan", "team_b": "Colombia", "score": None}
    ],
    "Jeudi 18 juin 2026": [
        {"team_a": "Czechia", "team_b": "South Africa", "score": None},
        {"team_a": "Switzerland", "team_b": "Bosnia and Herzegovina", "score": None},
        {"team_a": "Canada", "team_b": "Qatar", "score": None, "host": "Canada"},
        {"team_a": "Mexico", "team_b": "South Korea", "score": None}
    ],
    "Vendredi 19 juin 2026": [
        {"team_a": "United States", "team_b": "Australia", "score": None, "host": "United States"},
        {"team_a": "Scotland", "team_b": "Morocco", "score": None},
        {"team_a": "Brazil", "team_b": "Haiti", "score": None},
        {"team_a": "Turkey", "team_b": "Paraguay", "score": None}
    ],
    "Samedi 20 juin 2026": [
        {"team_a": "Netherlands", "team_b": "Sweden", "score": None},
        {"team_a": "Germany", "team_b": "Ivory Coast", "score": None},
        {"team_a": "Ecuador", "team_b": "Curaçao", "score": None},
        {"team_a": "Tunisia", "team_b": "Japan", "score": None}
    ],
    "Dimanche 21 juin 2026": [
        {"team_a": "Spain", "team_b": "Saudi Arabia", "score": None},
        {"team_a": "Belgium", "team_b": "Iran", "score": None},
        {"team_a": "Uruguay", "team_b": "Cape Verde", "score": None},
        {"team_a": "New Zealand", "team_b": "Egypt", "score": None}
    ],
    "Lundi 22 juin 2026": [
        {"team_a": "Argentina", "team_b": "Austria", "score": None},
        {"team_a": "France", "team_b": "Iraq", "score": None},
        {"team_a": "Norway", "team_b": "Senegal", "score": None},
        {"team_a": "Jordan", "team_b": "Algeria", "score": None}
    ],
    "Mardi 23 juin 2026": [
        {"team_a": "Portugal", "team_b": "Uzbekistan", "score": None},
        {"team_a": "England", "team_b": "Ghana", "score": None},
        {"team_a": "Panama", "team_b": "Croatia", "score": None},
        {"team_a": "Colombia", "team_b": "DR Congo", "score": None}
    ],
    "Mercredi 24 juin 2026": [
        {"team_a": "Switzerland", "team_b": "Canada", "score": None, "host": "Canada"},
        {"team_a": "Bosnia and Herzegovina", "team_b": "Qatar", "score": None},
        {"team_a": "Scotland", "team_b": "Brazil", "score": None},
        {"team_a": "Morocco", "team_b": "Haiti", "score": None},
        {"team_a": "Czechia", "team_b": "Mexico", "score": None},
        {"team_a": "South Africa", "team_b": "South Korea", "score": None}
    ],
    "Jeudi 25 juin 2026": [
        {"team_a": "Ecuador", "team_b": "Germany", "score": None},
        {"team_a": "Curaçao", "team_b": "Ivory Coast", "score": None},
        {"team_a": "Tunisia", "team_b": "Netherlands", "score": None},
        {"team_a": "Japan", "team_b": "Sweden", "score": None},
        {"team_a": "Turkey", "team_b": "United States", "score": None, "host": "United States"},
        {"team_a": "Paraguay", "team_b": "Australia", "score": None}
    ],
    "Vendredi 26 juin 2026": [
        {"team_a": "Norway", "team_b": "France", "score": None},
        {"team_a": "Senegal", "team_b": "Iraq", "score": None},
        {"team_a": "Uruguay", "team_b": "Spain", "score": None},
        {"team_a": "Cape Verde", "team_b": "Saudi Arabia", "score": None},
        {"team_a": "New Zealand", "team_b": "Belgium", "score": None},
        {"team_a": "Egypt", "team_b": "Iran", "score": None}
    ],
    "Samedi 27 juin 2026": [
        {"team_a": "Panama", "team_b": "England", "score": None},
        {"team_a": "Croatia", "team_b": "Ghana", "score": None},
        {"team_a": "Colombia", "team_b": "Portugal", "score": None},
        {"team_a": "DR Congo", "team_b": "Uzbekistan", "score": None},
        {"team_a": "Jordan", "team_b": "Argentina", "score": None},
        {"team_a": "Algeria", "team_b": "Austria", "score": None}
    ]
}

POPULAR_SCORES = {(1,0):1.0, (0,1):1.0, (1,1):1.0, (2,1):0.8, (1,2):0.8, (2,0):0.7, (0,2):0.7, (0,0):0.7}
ALPHA = 0.70

# ==========================================
# 2. LOGIQUE STATISTIQUE (AVEC FORME ET DOMICILE)
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

def calculate_lambdas(stat_a, stat_b, power_df, team_a_name, team_b_name, host_nation):
    off_a = stat_a.get('AVG_GOALS_10', 1.5)
    off_b = stat_b.get('AVG_GOALS_10', 1.5)
    def_a = stat_a.get('AVG_OPP_GOALS_10', 1.0)
    def_b = stat_b.get('AVG_OPP_GOALS_10', 1.0)
    
    xg_a = max(0.1, off_a * def_b)
    xg_b = max(0.1, off_b * def_a)
    
    elo_a = stat_a.get('ELO_POST', 1500)
    elo_b = stat_b.get('ELO_POST', 1500)
    
    # 1. BONUS DOMICILE (Host Advantage : +100 ELO)
    if host_nation == team_a_name:
        elo_a += 100
    elif host_nation == team_b_name:
        elo_b += 100
        
    # 2. BONUS DE FORME (Squad Power Index)
    if power_df is not None:
        power_a = power_df[power_df['Team'] == team_a_name]['Squad_Power_Index'].values
        power_b = power_df[power_df['Team'] == team_b_name]['Squad_Power_Index'].values
        
        p_a = power_a[0] if len(power_a) > 0 else 70.0
        p_b = power_b[0] if len(power_b) > 0 else 70.0
        
        # Chaque point de Power d'écart vaut 3 points ELO
        power_diff = p_a - p_b
        elo_a += power_diff * 3

    elo_diff = elo_a - elo_b
    
    if elo_diff > 0:
        xg_a += (elo_diff / 400) * 0.5
        xg_b -= (elo_diff / 400) * 0.2
    else:
        xg_b += (abs(elo_diff) / 400) * 0.5
        xg_a -= (abs(elo_diff) / 400) * 0.2
        
    return max(0.1, xg_a), max(0.1, xg_b), elo_a, elo_b

stats_df, power_df = load_football_data()
if stats_df is None:
    st.error("❌ Base de données introuvable. Exécute tes scripts de pipeline de données.")
    st.stop()

valid_teams = set(stats_df['TEAM_ID'].unique())

# ==========================================
# 3. INTERFACE UTILISATEUR MPP
# ==========================================
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    date_selected = st.selectbox("📅 Sélectionner la journée", list(CALENDRIER.keys()))
with col_header_2:
    mode = st.radio("🎯 Mode Stratégique", ["EV Max (Points Sécurisés)", "Challenger (Différenciation)"])

st.markdown("---")

matchs_du_jour = CALENDRIER[date_selected]
analyzed_matches = []

for idx, match in enumerate(matchs_du_jour):
    ta = match['team_a']
    tb = match['team_b']
    
    # GESTION DES MATCHS DÉJÀ TERMINÉS
    if match.get('score'):
        with st.container():
            st.success(f"✅ **Terminé :** {ta} **{match['score']}** {tb}")
            st.markdown("<br>", unsafe_allow_html=True)
        continue 

    if ta not in valid_teams or tb not in valid_teams:
        st.error(f"⚠️ Équipe non trouvée dans la base : {ta} ou {tb}.")
        continue
        
    stat_a = stats_df[stats_df['TEAM_ID'] == ta].iloc[0]
    stat_b = stats_df[stats_df['TEAM_ID'] == tb].iloc[0]
    
    la, lb, elo_a, elo_b = calculate_lambdas(stat_a, stat_b, power_df, ta, tb, match.get('host', ''))
    
    # Génération de la Matrice de Poisson (Calcul des vraies probabilités)
    matrix = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            matrix[i][j] = poisson.pmf(i, la) * poisson.pmf(j, lb)
            
    # Probabilités réelles de l'Oracle (Somme des probas des scores)
    prob_oracle_a = np.sum(np.tril(matrix, -1))
    prob_oracle_draw = np.trace(matrix)
    prob_oracle_b = np.sum(np.triu(matrix, 1))
    
    # Pré-remplissage des points MPP (Valeur indicative d'environ 100 pour un coin-flip)
    default_pts_a = int(35 / prob_oracle_a) if prob_oracle_a > 0 else 200
    default_pts_draw = int(35 / prob_oracle_draw) if prob_oracle_draw > 0 else 200
    default_pts_b = int(35 / prob_oracle_b) if prob_oracle_b > 0 else 200

    with st.expander(f"⚽ {ta} vs {tb} (ELO : {int(elo_a)} - {int(elo_b)})", expanded=True):
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2.5])
        
        with c1:
            st.markdown("**Points MPP (Gains)**")
            pts_a = st.number_input(f"Pts MPP {ta}", value=default_pts_a, key=f"ca_{idx}", step=1)
            pts_draw = st.number_input(f"Pts MPP Nul", value=default_pts_draw, key=f"cd_{idx}", step=1)
            pts_b = st.number_input(f"Pts MPP {tb}", value=default_pts_b, key=f"cb_{idx}", step=1)
            
        with c2:
            st.markdown("**Choix de la foule (%)**")
            foule_a = st.slider(f"% sur {ta}", 0, 100, int(prob_oracle_a * 100), key=f"fa_{idx}")
            foule_draw = st.slider(f"% sur Nul", 0, 100, int(prob_oracle_draw * 100), key=f"fd_{idx}")
            foule_b = st.slider(f"% sur {tb}", 0, 100, int(prob_oracle_b * 100), key=f"fb_{idx}")

        # L'Équation d'Espérance (Expected Points MPP)
        ev_dict = {
            'a': prob_oracle_a * pts_a,
            'draw': prob_oracle_draw * pts_draw,
            'b': prob_oracle_b * pts_b
        }
        
        fav_ev = max(ev_dict, key=ev_dict.get)
        
        # Levier Contrarien : L'EV divisée par le % de gens qui la jouent
        crowd_dict = {'a': max(foule_a/100, 0.01), 'draw': max(foule_draw/100, 0.01), 'b': max(foule_b/100, 0.01)}
        leverage_dict = {k: ev_dict[k] / crowd_dict[k] for k in ev_dict}
        
        if mode == "Challenger (Différenciation)":
            pick = max((k for k in leverage_dict if k != fav_ev), key=lambda k: leverage_dict[k])
            why = "Levier Contrarien"
        else:
            pick = fav_ev
            why = "Max Expected Points"
            
        # Détermination du score
        def _ok(outcome, i, j):
            return (outcome=='a' and i>j) or (outcome=='draw' and i==j) or (outcome=='b' and i<j)
            
        best_score, max_p = (0,0), -1.0
        rare_score, max_r = (0,0), -1.0
        
        for i in range(7):
            for j in range(7):
                if _ok(pick, i, j):
                    if matrix[i][j] > max_p:
                        max_p, best_score = matrix[i][j], (i, j)
                    adj = matrix[i][j] * (1 - ALPHA * POPULAR_SCORES.get((i, j), 0.2))
                    if adj > max_r:
                        max_r, rare_score = adj, (i, j)

        labels = {'a': ta, 'draw': 'Match Nul', 'b': tb}
        
        with c3:
            st.markdown("**Oracle Analytics**")
            st.metric("PRONO MPP À COCHER", labels[pick].upper())
            st.write(f"Espérance : **{ev_dict[pick]:.1f} pts**")
            st.caption(f"Logique : {why}")
            
        with c4:
            st.markdown("**Simulation du Score (Bonus x3)**")
            st.info(f"🏆 **Score Standard : {best_score[0]} - {best_score[1]}** (Proba: {max_p:.1%})")
            st.success(f"⭐ **Score Rareté : {rare_score[0]} - {rare_score[1]}** (Pour maximiser l'écart)")

        analyzed_matches.append({
            'match': f"{ta} vs {tb}",
            'pick': labels[pick],
            'leverage': leverage_dict[pick]
        })

# ==========================================
# 4. LE RECOMMANDATEUR DE BONUS X2
# ==========================================
if analyzed_matches:
    st.markdown("---")
    st.header("♟️ Radar du Bonus x2")
    best_x2 = max(analyzed_matches, key=lambda x: x['leverage'])
    st.success(f"🚨 **LEVIER MAXIMAL DETECTÉ** : Si tu dois utiliser ton bonus x2 aujourd'hui, coche **{best_x2['pick'].upper()}** sur le match **{best_x2['match']}**. Mathématiquement, c'est là que l'écart entre la probabilité de l'algorithme, les points offerts, et l'ignorance de la foule est le plus grand.")