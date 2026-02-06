"""
NBA Oracle - Application Streamlit Cloud
=========================================

Cette version est optimisée pour le déploiement cloud:
- Chargement des données depuis le repo GitHub
- Cache intelligent pour les performances
- Gestion des erreurs robuste
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import sys

# Ajouter src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import du collecteur de cotes (optionnel)
try:
    from src.odds_collector import OddsCollector
    ODDS_AVAILABLE = True
except ImportError:
    ODDS_AVAILABLE = False

# Import NBA API
try:
    from nba_api.stats.endpoints import scoreboardv2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🏀 NBA Oracle",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
MODEL_PATH = 'models/nba_model.pkl'
DATA_PATH = 'data/processed_games.csv'
FEATURES_PATH = 'models/features.txt'
METRICS_PATH = 'models/metrics.txt'


# =============================================================================
# FONCTIONS DE CHARGEMENT (avec cache)
# =============================================================================

@st.cache_resource
def load_model():
    """Charge le modèle (cache permanent)."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Erreur chargement modèle: {e}")
            return None
    return None


@st.cache_data(ttl=3600)  # Cache 1 heure
def load_team_stats():
    """Charge les stats des équipes."""
    if not os.path.exists(DATA_PATH):
        return None, None
    
    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Dernières stats par équipe
    latest = df.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1)
    latest = latest.set_index('TEAM_ID')
    
    # Dernière date de match
    last_dates = df.groupby('TEAM_ID')['GAME_DATE'].max()
    
    return latest, last_dates


@st.cache_data(ttl=300)  # Cache 5 minutes
def get_today_games():
    """Récupère les matchs du jour via NBA API."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    try:
        today = datetime.now()
        board = scoreboardv2.ScoreboardV2(
            game_date=today.strftime('%Y-%m-%d'),
            timeout=30
        )
        return board.game_header.get_data_frame()
    except Exception as e:
        st.warning(f"Impossible de récupérer les matchs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache 30 minutes
def get_odds():
    """Récupère les cotes."""
    if not ODDS_AVAILABLE:
        return None
    
    try:
        collector = OddsCollector()
        odds = collector.get_upcoming_odds()
        return odds if not odds.empty else None
    except:
        return None


def load_features():
    """Charge la liste des features."""
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return ['ELO_PRE_DIFF', 'REST_DAYS_DIFF']


def load_metrics():
    """Charge les métriques du modèle."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return f.read()
    return "Métriques non disponibles"


# =============================================================================
# PRÉDICTION
# =============================================================================

def predict_match(model, home_stats, away_stats, home_rest, away_rest, features_list):
    """Prédit le résultat d'un match."""
    
    features = {}
    
    # Construire les features
    feature_mapping = {
        'ELO_PRE_DIFF': ('ELO_PRE', 'ELO_PRE', 1500),
        'REST_DAYS_DIFF': (None, None, None),  # Calculé différemment
        'STREAK_DIFF': ('STREAK', 'STREAK', 0),
        'FORM_5_DIFF': ('FORM_5', 'FORM_5', 0.5),
        'PTS_DIFF': ('AVG_PTS_10', 'AVG_PTS_10', 110),
        'EFG_PCT_DIFF': ('AVG_EFG_PCT_10', 'AVG_EFG_PCT_10', 0.5),
        'TOV_PCT_DIFF': ('AVG_TOV_PCT_10', 'AVG_TOV_PCT_10', 0.13),
        'NET_RTG_DIFF': ('AVG_NET_RTG_10', 'AVG_NET_RTG_10', 0),
    }
    
    for feat in features_list:
        if feat == 'REST_DAYS_DIFF':
            features[feat] = min(home_rest, 7) - min(away_rest, 7)
        elif feat == 'HOME_IS_B2B':
            features[feat] = 1 if home_rest <= 1 else 0
        elif feat == 'AWAY_IS_B2B':
            features[feat] = 1 if away_rest <= 1 else 0
        elif feat == 'HOME_REST_DAYS':
            features[feat] = min(home_rest, 7)
        elif feat == 'AWAY_REST_DAYS':
            features[feat] = min(away_rest, 7)
        elif feat in feature_mapping:
            home_col, away_col, default = feature_mapping[feat]
            if home_col:
                home_val = home_stats.get(home_col, default)
                away_val = away_stats.get(away_col, default)
                features[feat] = home_val - away_val
        else:
            features[feat] = 0
    
    # Prédiction
    X = pd.DataFrame([{f: features.get(f, 0) for f in features_list}])
    prob_home = model.predict_proba(X)[0][1]
    
    return prob_home, features


def match_odds(odds_df, home_name, away_name):
    """Trouve les cotes pour un match."""
    if odds_df is None or odds_df.empty:
        return None, None
    
    for _, row in odds_df.iterrows():
        h_match = home_name.lower() in row['home_team'].lower() or \
                  row['home_team'].lower() in home_name.lower() or \
                  home_name.split()[-1].lower() in row['home_team'].lower()
        
        a_match = away_name.lower() in row['away_team'].lower() or \
                  row['away_team'].lower() in away_name.lower() or \
                  away_name.split()[-1].lower() in row['away_team'].lower()
        
        if h_match and a_match:
            return row['implied_prob_home'], row['implied_prob_away']
    
    return None, None


# =============================================================================
# INTERFACE
# =============================================================================

def main():
    # Header
    st.title("🏀 NBA Oracle")
    st.markdown(f"**Prédictions du {datetime.now().strftime('%A %d %B %Y')}**")
    
    # Chargement
    model = load_model()
    team_stats, last_dates = load_team_stats()
    features_list = load_features()
    games = get_today_games()
    odds_data = get_odds()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Informations")
        
        # Métriques
        st.subheader("🎯 Performance du modèle")
        metrics = load_metrics()
        st.code(metrics)
        
        # Dernière mise à jour
        if os.path.exists(DATA_PATH):
            mtime = os.path.getmtime(DATA_PATH)
            last_update = datetime.fromtimestamp(mtime)
            st.info(f"🔄 Dernière MAJ: {last_update.strftime('%d/%m/%Y %H:%M')}")
        
        # Status cotes
        st.subheader("🎰 Cotes")
        if odds_data is not None:
            st.success(f"✅ {len(odds_data)} matchs")
        else:
            st.warning("⚠️ Non disponibles")
        
        # Légende
        st.markdown("---")
        st.caption("🤖 = Notre prédiction")
        st.caption("🎰 = Cotes du marché")
        st.caption("💎 = Value bet (edge >5%)")
    
    # Vérifications
    if model is None:
        st.error("❌ Modèle non trouvé. Le pipeline doit être exécuté.")
        st.stop()
    
    if team_stats is None:
        st.error("❌ Données non trouvées.")
        st.stop()
    
    if games.empty:
        st.warning("📅 Aucun match NBA programmé aujourd'hui.")
        st.info("Reviens demain pour les prédictions !")
        st.stop()
    
    # Affichage des matchs
    st.success(f"✅ {len(games)} match(s) aujourd'hui")
    st.markdown("---")
    
    for _, game in games.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        if home_id not in team_stats.index or away_id not in team_stats.index:
            continue
        
        home_data = team_stats.loc[home_id]
        away_data = team_stats.loc[away_id]
        
        home_name = home_data.get('TEAM_NAME', f'Team {home_id}')
        away_name = away_data.get('TEAM_NAME', f'Team {away_id}')
        
        # Repos
        today = datetime.now()
        home_rest = (today - last_dates[home_id]).days
        away_rest = (today - last_dates[away_id]).days
        
        # Prédiction
        prob_home, used_features = predict_match(
            model, home_data.to_dict(), away_data.to_dict(),
            home_rest, away_rest, features_list
        )
        prob_away = 1 - prob_home
        
        # Cotes
        market_home, market_away = match_odds(odds_data, home_name, away_name)
        has_odds = market_home is not None
        
        # Value bet
        edge = None
        if has_odds:
            edge = prob_home - market_home
        
        # --- Affichage ---
        col1, col2, col3 = st.columns([2, 1.5, 2])
        
        with col1:
            st.markdown(f"### 🏃 {away_name}")
            m1, m2 = st.columns(2)
            m1.metric("ELO", int(away_data.get('ELO_PRE', 1500)))
            m2.metric("Repos", f"{away_rest}j", 
                     delta="B2B!" if away_rest <= 1 else None,
                     delta_color="inverse")
        
        with col2:
            # Prédiction
            winner = home_name if prob_home > 0.5 else away_name
            conf = max(prob_home, prob_away)
            
            color = "#28a745" if conf >= 0.65 else "#ffc107" if conf >= 0.55 else "#6c757d"
            
            st.markdown(f"""
                <div style='text-align: center; background: #1a1a2e; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                    <p style='margin: 0; color: #888; font-size: 12px;'>🤖 PRÉDICTION</p>
                    <h2 style='color: {color}; margin: 5px 0;'>{conf:.0%}</h2>
                    <p style='margin: 0;'>→ {winner.split()[-1]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Cotes + Value
            if has_odds:
                market_winner = home_name if market_home > market_away else away_name
                st.markdown(f"""
                    <div style='text-align: center; background: #2a2a3e; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                        <p style='margin: 0; color: #888; font-size: 11px;'>🎰 MARCHÉ: {max(market_home, market_away):.0%} → {market_winner.split()[-1]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if abs(edge) >= 0.05:
                    value_team = home_name.split()[-1] if edge > 0 else away_name.split()[-1]
                    value_color = "#28a745" if abs(edge) >= 0.10 else "#ffc107"
                    st.markdown(f"""
                        <div style='text-align: center; background: {value_color}22; border: 1px solid {value_color}; padding: 5px; border-radius: 5px; margin-top: 5px;'>
                            <span style='color: {value_color}; font-size: 12px;'>💎 VALUE: {value_team} +{abs(edge):.1%}</span>
                        </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"### 🏠 {home_name}")
            m1, m2 = st.columns(2)
            home_elo = int(home_data.get('ELO_PRE', 1500))
            away_elo = int(away_data.get('ELO_PRE', 1500))
            m1.metric("ELO", home_elo, delta=home_elo - away_elo)
            m2.metric("Repos", f"{home_rest}j",
                     delta="B2B!" if home_rest <= 1 else None,
                     delta_color="inverse")
        
        st.markdown("---")
    
    # Footer
    st.caption("⚠️ Ce système est à but éducatif. Les paris sportifs comportent des risques.")
    st.caption(f"🔄 Mise à jour automatique quotidienne via GitHub Actions")


if __name__ == "__main__":
    main()
