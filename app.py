import streamlit as st 
import pandas as pd 
import numpy as np
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NBA Oracle", 
    layout="wide", 
    page_icon="🏀"
)

# Chemins des fichiers
MODEL_PATHS = ['models/nba_model.pkl', 'nba_model.pkl']
DATA_PATHS = ['data/processed_games.csv', 'processed_games.csv']
FEATURES_PATH = 'models/features.txt'


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def load_model():
    """Charge le modèle de prédiction."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                return None, f"Erreur: {e}"
    return None, "Modèle introuvable"


def load_features_list():
    """Charge la liste des features utilisées par le modèle."""
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None


@st.cache_data(ttl=3600)
def load_team_stats():
    """Charge les dernières stats de chaque équipe."""
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Dernières stats par équipe
            latest = df.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1)
            latest = latest.set_index('TEAM_ID')
            
            # Dernière date de match par équipe
            last_dates = df.groupby('TEAM_ID')['GAME_DATE'].max()
            
            return latest, last_dates
    
    return None, None


@st.cache_data(ttl=600)  # Cache 10 minutes
def get_today_schedule():
    """Récupère les matchs du jour."""
    try:
        today = datetime.now()
        board = scoreboardv2.ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
        return board.game_header.get_data_frame()
    except Exception as e:
        st.error(f"Erreur API NBA: {e}")
        return pd.DataFrame()


# =============================================================================
# FONCTION DE PRÉDICTION
# =============================================================================

def predict_matchup(model, home_stats, away_stats, home_rest, away_rest, features_list):
    """
    Prédit le résultat d'un match en construisant les features correctement.
    
    Retourne: (probabilité victoire domicile, features utilisées)
    """
    # Construire le vecteur de features
    features = {}
    
    # --- Features différentielles ---
    if 'ELO_PRE_DIFF' in features_list:
        features['ELO_PRE_DIFF'] = home_stats.get('ELO_PRE', 1500) - away_stats.get('ELO_PRE', 1500)
    
    if 'REST_DAYS_DIFF' in features_list:
        features['REST_DAYS_DIFF'] = min(home_rest, 7) - min(away_rest, 7)
    
    if 'STREAK_DIFF' in features_list:
        features['STREAK_DIFF'] = home_stats.get('STREAK', 0) - away_stats.get('STREAK', 0)
    
    if 'FORM_5_DIFF' in features_list:
        features['FORM_5_DIFF'] = home_stats.get('FORM_5', 0.5) - away_stats.get('FORM_5', 0.5)
    
    if 'PTS_DIFF' in features_list:
        features['PTS_DIFF'] = home_stats.get('AVG_PTS_10', 110) - away_stats.get('AVG_PTS_10', 110)
    
    if 'EFG_PCT_DIFF' in features_list:
        features['EFG_PCT_DIFF'] = home_stats.get('AVG_EFG_PCT_10', 0.5) - away_stats.get('AVG_EFG_PCT_10', 0.5)
    
    if 'TOV_PCT_DIFF' in features_list:
        features['TOV_PCT_DIFF'] = home_stats.get('AVG_TOV_PCT_10', 0.13) - away_stats.get('AVG_TOV_PCT_10', 0.13)
    
    if 'NET_RTG_DIFF' in features_list:
        features['NET_RTG_DIFF'] = home_stats.get('AVG_NET_RTG_10', 0) - away_stats.get('AVG_NET_RTG_10', 0)
    
    # --- Features contextuelles ---
    if 'HOME_IS_B2B' in features_list:
        features['HOME_IS_B2B'] = 1 if home_rest <= 1 else 0
    
    if 'AWAY_IS_B2B' in features_list:
        features['AWAY_IS_B2B'] = 1 if away_rest <= 1 else 0
    
    if 'HOME_REST_DAYS' in features_list:
        features['HOME_REST_DAYS'] = min(home_rest, 7)
    
    if 'AWAY_REST_DAYS' in features_list:
        features['AWAY_REST_DAYS'] = min(away_rest, 7)
    
    # --- Features absolues (fallback) ---
    for prefix in ['HOME_', 'AWAY_']:
        stats = home_stats if prefix == 'HOME_' else away_stats
        
        for col in ['ELO_PRE', 'AVG_PTS_10', 'AVG_EFG_PCT_10', 'AVG_TOV_PCT_10', 
                    'AVG_OREB_PCT_10', 'AVG_FT_RATE_10', 'AVG_NET_RTG_10']:
            feat_name = f'{prefix}{col}'
            if feat_name in features_list:
                features[feat_name] = stats.get(col, 0)
    
    # Construire le DataFrame avec les features dans le bon ordre
    X = pd.DataFrame([{f: features.get(f, 0) for f in features_list}])
    
    # Prédiction
    prob_home = model.predict_proba(X)[0][1]
    
    return prob_home, features


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    st.title("🏀 NBA Oracle : Prédictions du Jour")
    st.markdown(f"**Date :** {datetime.now().strftime('%A %d %B %Y')}")
    
    # --- Chargement ---
    with st.spinner("Chargement du système..."):
        model, model_msg = load_model()
        features_list = load_features_list()
        team_stats, last_dates = load_team_stats()
        schedule = get_today_schedule()
    
    # --- Vérifications ---
    if model is None:
        st.error(f"🚨 Modèle introuvable: {model_msg}")
        st.info("👉 Lance: `python src/train_model.py`")
        st.stop()
    
    if team_stats is None:
        st.error("🚨 Données manquantes")
        st.info("👉 Lance: `python src/process_data.py`")
        st.stop()
    
    if features_list is None:
        st.warning("⚠️ Liste des features non trouvée, utilisation des défauts")
        features_list = ['ELO_PRE_DIFF', 'REST_DAYS_DIFF', 'HOME_REST_DAYS', 'AWAY_REST_DAYS']
    
    if schedule.empty:
        st.warning("⚠️ Aucun match programmé aujourd'hui")
        st.stop()
    
    st.success(f"✅ Système chargé | {len(schedule)} matchs | Modèle: {model_msg}")
    
    # --- Sidebar avec infos ---
    with st.sidebar:
        st.header("📊 Informations")
        
        # Métriques du modèle
        if os.path.exists('models/metrics.txt'):
            with open('models/metrics.txt', 'r') as f:
                st.text(f.read())
        
        st.markdown("---")
        st.caption(f"Features utilisées: {len(features_list)}")
        with st.expander("Voir les features"):
            for f in features_list:
                st.caption(f"• {f}")
    
    # --- Affichage des matchs ---
    st.markdown("---")
    
    for idx, game in schedule.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        # Vérifier que les équipes sont dans nos données
        if home_id not in team_stats.index or away_id not in team_stats.index:
            st.warning(f"⚠️ Données manquantes pour un match")
            continue
        
        home_data = team_stats.loc[home_id]
        away_data = team_stats.loc[away_id]
        
        # Calculer le repos
        today = datetime.now()
        home_rest = (today - last_dates[home_id]).days
        away_rest = (today - last_dates[away_id]).days
        
        # Prédiction
        prob_home, used_features = predict_matchup(
            model, 
            home_data.to_dict(), 
            away_data.to_dict(),
            home_rest, 
            away_rest,
            features_list
        )
        
        # --- Interface du match ---
        st.markdown("### " + "─" * 30)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        # Équipe extérieure (gauche)
        with col1:
            away_name = away_data.get('TEAM_NAME', f'Team {away_id}')
            away_elo = int(away_data.get('ELO_PRE', 1500))
            away_form = away_data.get('FORM_5', 0.5)
            
            st.markdown(f"### 🏃 {away_name}")
            
            # Métriques
            m1, m2, m3 = st.columns(3)
            m1.metric("ELO", away_elo)
            m2.metric("Repos", f"{away_rest}j", delta="B2B" if away_rest <= 1 else None, delta_color="inverse")
            m3.metric("Forme", f"{away_form:.0%}" if pd.notna(away_form) else "N/A")
            
            # Barre de probabilité
            prob_away = 1 - prob_home
            if prob_away > 0.5:
                st.progress(prob_away, text=f"**{prob_away:.1%}**")
        
        # Prédiction (centre)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Déterminer le vainqueur prédit
            if prob_home > 0.5:
                winner = home_data.get('TEAM_NAME', 'Domicile')
                confidence = prob_home
            else:
                winner = away_data.get('TEAM_NAME', 'Extérieur')
                confidence = 1 - prob_home
            
            # Couleur selon confiance
            if confidence >= 0.65:
                color = "green"
                emoji = "🔥"
            elif confidence >= 0.55:
                color = "orange"
                emoji = "👍"
            else:
                color = "gray"
                emoji = "🤔"
            
            st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <h1 style='color: {color}; margin: 0;'>{confidence:.0%}</h1>
                    <p style='font-size: 14px; margin: 5px 0;'>{emoji} {winner}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Équipe domicile (droite)
        with col3:
            home_name = home_data.get('TEAM_NAME', f'Team {home_id}')
            home_elo = int(home_data.get('ELO_PRE', 1500))
            home_form = home_data.get('FORM_5', 0.5)
            
            st.markdown(f"### 🏠 {home_name}")
            
            # Métriques
            m1, m2, m3 = st.columns(3)
            m1.metric("ELO", home_elo, delta=home_elo - away_elo)
            m2.metric("Repos", f"{home_rest}j", delta="B2B" if home_rest <= 1 else None, delta_color="inverse")
            m3.metric("Forme", f"{home_form:.0%}" if pd.notna(home_form) else "N/A")
            
            # Barre de probabilité
            if prob_home >= 0.5:
                st.progress(prob_home, text=f"**{prob_home:.1%}**")
        
        # Détails (expander)
        with st.expander("📊 Détails de la prédiction"):
            detail_cols = st.columns(2)
            
            with detail_cols[0]:
                st.caption("**Features utilisées:**")
                for feat, val in used_features.items():
                    if 'DIFF' in feat:
                        st.caption(f"• {feat}: {val:+.2f}")
            
            with detail_cols[1]:
                st.caption("**Interprétation:**")
                elo_diff = used_features.get('ELO_PRE_DIFF', 0)
                if elo_diff > 50:
                    st.caption("✅ Domicile nettement favori (ELO)")
                elif elo_diff < -50:
                    st.caption("✅ Extérieur nettement favori (ELO)")
                else:
                    st.caption("⚖️ Match équilibré")
                
                rest_diff = used_features.get('REST_DAYS_DIFF', 0)
                if rest_diff >= 2:
                    st.caption("✅ Avantage repos domicile")
                elif rest_diff <= -2:
                    st.caption("❌ Désavantage repos domicile")
    
    # --- Footer ---
    st.markdown("---")
    st.caption("🏀 NBA Oracle | Prédictions basées sur l'ELO, la forme récente et le contexte")


if __name__ == "__main__":
    main()