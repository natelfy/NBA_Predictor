import streamlit as st 
import pandas as pd 
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import os

# CONFIGURATION DE LA PAGE
st.set_page_config(page_title="NBA Oracle", layout="wide", page_icon="🏀")

# --- FONCTION : CHARGEMENT ROBUSTE DU MODÈLE ---
def load_model_robust():
    # Liste des endroits probables où ton modèle peut se cacher
    possible_paths = [
        'models/nba_model.pkl',
        'nba_model.pkl',
        'NBA_Predictor/models/nba_model.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                return None, f"Erreur lecture {path}: {e}"
    
    return None, "Fichier introuvable"

# --- FONCTION : CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=3600)
def get_data_and_schedule():
    # 1. Charger Stats (Pareil, on cherche le fichier partout)
    data_paths = ['data/processed_games.csv', 'processed_games.csv']
    df = None
    for p in data_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
            
    if df is None:
        return None, None, None

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    # Dernières stats connues
    latest_stats = df.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1).set_index('TEAM_ID')
    # Dates des derniers matchs
    last_game_dates = df.groupby('TEAM_ID')['GAME_DATE'].max()

    # 2. Charger Calendrier (Aujourd'hui)
    today = datetime.now()
    game_date_str = today.strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=game_date_str)
        schedule = board.game_header.get_data_frame()
    except:
        schedule = pd.DataFrame()

    return latest_stats, last_game_dates, schedule

# --- INTERFACE PRINCIPALE ---
st.title("🏀 NBA Oracle : Prédictions du Jour")
st.markdown(f"**Date :** {datetime.now().strftime('%d/%m/%Y')}")

# 1. Chargement Modèle
model, msg = load_model_robust()
if model is None:
    st.error(f"🚨 CRITICAL ERROR : Le modèle est introuvable.\nDétail : {msg}")
    st.info("👉 Solution : Lance 'python train_model.py' pour générer le fichier 'models/nba_model.pkl'.")
    st.stop() # On arrête tout ici si pas de modèle

# 2. Chargement Data
stats, dates, schedule = get_data_and_schedule()

if stats is None:
    st.error("🚨 CRITICAL ERROR : Données manquantes (processed_games.csv).")
    st.info("👉 Solution : Lance 'python process_data.py'.")
    st.stop()

if schedule is None or schedule.empty:
    st.warning("⚠️ Aucun match trouvé pour cette nuit (ou API inaccessible).")
    st.stop()

st.success(f"✅ Système chargé. {len(schedule)} matchs détectés.")

# --- AFFICHAGE DES MATCHS ---
# On n'utilise PAS de st.form ou st.button pour éviter les resets intempestifs
for index, game in schedule.iterrows():
    h_id, a_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']

    if h_id in stats.index and a_id in stats.index:
        s_h, s_a = stats.loc[h_id], stats.loc[a_id]
        
        # Calcul Fatigue
        today_dt = datetime.now()
        rest_h = (today_dt - dates[h_id]).days
        rest_a = (today_dt - dates[a_id]).days
        
        # UI
        st.markdown("---")
        c1, c2, c3, c4 = st.columns([3, 2, 3, 2])

        # Paramètres interactifs (Checkboxes)
        # L'état est conservé grâce aux clés uniques
        with c4:
            st.caption("Paramètres")
            star_h = st.checkbox("Absent Majeur", key=f"sh_{index}")
            star_a = st.checkbox("Absent Majeur", key=f"sa_{index}")

        # Vecteurs (Même ordre que train_model.py !)
        # Features: [IS_HOME, REST_DAYS, ELO_PRE, AVG_EFG_PCT_10, AVG_TOV_PCT_10, AVG_FT_RATE_10, AVG_OREB_PCT_10, AVG_PTS_10]
        f_h = [[1, min(rest_h,7), s_h['ELO_PRE'], s_h['AVG_EFG_PCT_10'], s_h['AVG_TOV_PCT_10'], s_h['AVG_FT_RATE_10'], s_h['AVG_OREB_PCT_10'], s_h['AVG_PTS_10']]]
        f_a = [[0, min(rest_a,7), s_a['ELO_PRE'], s_a['AVG_EFG_PCT_10'], s_a['AVG_TOV_PCT_10'], s_a['AVG_FT_RATE_10'], s_a['AVG_OREB_PCT_10'], s_a['AVG_PTS_10']]]

        # Prédiction
        prob_h = model.predict_proba(f_h)[0][1]
        prob_a = model.predict_proba(f_a)[0][1]

        # Ajustement Humain
        final_p = (prob_h + (1 - prob_a)) / 2
        if star_h: final_p -= 0.15
        if star_a: final_p += 0.15
        final_p = max(0.01, min(0.99, final_p))

        # Affichage
        name_h = s_h['TEAM_NAME']
        name_a = s_a['TEAM_NAME']

        with c1:
            st.markdown(f"<h3 style='text-align: right'>{name_a}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right'>Elo {int(s_a['ELO_PRE'])}</p>", unsafe_allow_html=True)
            if final_p < 0.5:
                st.progress(1 - final_p)

        with c2:
            conf = max(final_p, 1-final_p)
            color = "green" if conf > 0.6 else "orange"
            st.markdown(f"<h4 style='text-align: center; color: {color}'>{conf:.1%}</h4>", unsafe_allow_html=True)
            if final_p > 0.5:
                st.markdown(f"<p style='text-align: center'>👉 {name_h}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align: center'>👉 {name_a}</p>", unsafe_allow_html=True)

        with c3:
            st.markdown(f"<h3 style='text-align: left'>{name_h}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left'>Elo {int(s_h['ELO_PRE'])}</p>", unsafe_allow_html=True)
            if final_p >= 0.5:
                st.progress(final_p)