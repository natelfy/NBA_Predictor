"""
NBA Oracle - Application avec Tracking
=======================================

Version avec suivi des prédictions intégré.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import sys

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports
try:
    from src.odds_collector import OddsCollector
    ODDS_AVAILABLE = True
except ImportError:
    ODDS_AVAILABLE = False

try:
    from src.tracker import PredictionTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from nba_api.stats.endpoints import scoreboardv2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="🏀 NBA Oracle",
    page_icon="🏀",
    layout="wide"
)

MODEL_PATH = 'models/nba_model.pkl'
DATA_PATH = 'data/processed_games.csv'
FEATURES_PATH = 'models/features.txt'


# =============================================================================
# CHARGEMENT
# =============================================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data(ttl=3600)
def load_team_stats():
    if not os.path.exists(DATA_PATH):
        return None, None
    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    latest = df.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1).set_index('TEAM_ID')
    last_dates = df.groupby('TEAM_ID')['GAME_DATE'].max()
    return latest, last_dates

@st.cache_data(ttl=300)
def get_today_games():
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    try:
        board = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime('%Y-%m-%d'), timeout=30)
        return board.game_header.get_data_frame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_odds():
    if not ODDS_AVAILABLE:
        return None
    try:
        collector = OddsCollector()
        odds = collector.get_upcoming_odds()
        return odds if not odds.empty else None
    except:
        return None

def load_features():
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'r') as f:
            return [l.strip() for l in f.readlines()]
    return ['ELO_PRE_DIFF', 'REST_DAYS_DIFF']


# =============================================================================
# PRÉDICTION
# =============================================================================

def predict_match(model, home_stats, away_stats, home_rest, away_rest, features_list):
    features = {}
    
    mapping = {
        'ELO_PRE_DIFF': ('ELO_PRE', 1500),
        'STREAK_DIFF': ('STREAK', 0),
        'FORM_5_DIFF': ('FORM_5', 0.5),
        'PTS_DIFF': ('AVG_PTS_10', 110),
        'EFG_PCT_DIFF': ('AVG_EFG_PCT_10', 0.5),
        'TOV_PCT_DIFF': ('AVG_TOV_PCT_10', 0.13),
        'NET_RTG_DIFF': ('AVG_NET_RTG_10', 0),
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
        elif feat in mapping:
            col, default = mapping[feat]
            features[feat] = home_stats.get(col, default) - away_stats.get(col, default)
        else:
            features[feat] = 0
    
    X = pd.DataFrame([{f: features.get(f, 0) for f in features_list}])
    return model.predict_proba(X)[0][1], features

def match_odds(odds_df, home_name, away_name):
    if odds_df is None:
        return None, None
    for _, row in odds_df.iterrows():
        h = home_name.lower() in row['home_team'].lower() or home_name.split()[-1].lower() in row['home_team'].lower()
        a = away_name.lower() in row['away_team'].lower() or away_name.split()[-1].lower() in row['away_team'].lower()
        if h and a:
            return row['implied_prob_home'], row['implied_prob_away']
    return None, None


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    # Header
    st.title("🏀 NBA Oracle")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Prédictions du jour", "📈 Performance (Tracking)"])
    
    # Chargement
    model = load_model()
    team_stats, last_dates = load_team_stats()
    features_list = load_features()
    games = get_today_games()
    odds_data = get_odds()
    
    # Tracker
    tracker = PredictionTracker() if TRACKER_AVAILABLE else None
    
    # =================================
    # TAB 1: PRÉDICTIONS
    # =================================
    with tab1:
        st.markdown(f"**{datetime.now().strftime('%A %d %B %Y')}**")
        
        if model is None:
            st.error("❌ Modèle non trouvé")
            st.stop()
        
        if team_stats is None:
            st.error("❌ Données non trouvées")
            st.stop()
        
        if games.empty:
            st.warning("📅 Aucun match aujourd'hui")
            st.stop()
        
        # Bouton pour sauvegarder les prédictions
        col_info, col_save = st.columns([3, 1])
        with col_info:
            st.success(f"✅ {len(games)} match(s) | {'🎰 Cotes dispo' if odds_data is not None else '⚠️ Pas de cotes'}")
        with col_save:
            if tracker and st.button("💾 Sauvegarder prédictions"):
                save_predictions_today(tracker, model, team_stats, last_dates, games, features_list, odds_data)
                st.success("✅ Prédictions sauvegardées!")
        
        st.markdown("---")
        
        # Afficher les matchs
        for _, game in games.iterrows():
            home_id = game['HOME_TEAM_ID']
            away_id = game['VISITOR_TEAM_ID']
            
            if home_id not in team_stats.index or away_id not in team_stats.index:
                continue
            
            home_data = team_stats.loc[home_id]
            away_data = team_stats.loc[away_id]
            home_name = home_data.get('TEAM_NAME', f'Team {home_id}')
            away_name = away_data.get('TEAM_NAME', f'Team {away_id}')
            
            home_rest = (datetime.now() - last_dates[home_id]).days
            away_rest = (datetime.now() - last_dates[away_id]).days
            
            prob_home, _ = predict_match(model, home_data.to_dict(), away_data.to_dict(), home_rest, away_rest, features_list)
            market_home, _ = match_odds(odds_data, home_name, away_name)
            
            # Affichage
            col1, col2, col3 = st.columns([2, 1.5, 2])
            
            with col1:
                st.markdown(f"### 🏃 {away_name}")
                st.metric("ELO", int(away_data.get('ELO_PRE', 1500)))
            
            with col2:
                winner = home_name if prob_home > 0.5 else away_name
                conf = max(prob_home, 1 - prob_home)
                color = "#28a745" if conf >= 0.65 else "#ffc107" if conf >= 0.55 else "#6c757d"
                
                st.markdown(f"""
                    <div style='text-align: center; background: #1a1a2e; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                        <p style='margin: 0; color: #888;'>🤖 PRÉDICTION</p>
                        <h2 style='color: {color}; margin: 5px 0;'>{conf:.0%}</h2>
                        <p>→ {winner.split()[-1]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if market_home:
                    edge = prob_home - market_home
                    if abs(edge) >= 0.05:
                        vt = home_name.split()[-1] if edge > 0 else away_name.split()[-1]
                        st.markdown(f"<div style='text-align:center;color:#28a745;'>💎 VALUE: {vt} +{abs(edge):.1%}</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"### 🏠 {home_name}")
                st.metric("ELO", int(home_data.get('ELO_PRE', 1500)))
            
            st.markdown("---")
    
    # =================================
    # TAB 2: TRACKING
    # =================================
    with tab2:
        st.header("📈 Performance du modèle")
        
        if not TRACKER_AVAILABLE:
            st.warning("⚠️ Module tracker non disponible")
            st.stop()
        
        if tracker is None:
            tracker = PredictionTracker()
        
        # Stats
        stats = tracker.get_stats(days=30)
        
        if stats['total_predictions'] == 0:
            st.info("📭 Aucune prédiction trackée pour l'instant.")
            st.markdown("""
            **Comment ça marche :**
            1. Clique sur "💾 Sauvegarder prédictions" dans l'onglet Prédictions
            2. Le lendemain, les résultats seront automatiquement récupérés
            3. Reviens ici pour voir ta vraie performance !
            """)
        else:
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Prédictions", stats['total_predictions'])
            col2.metric("✅ Correctes", stats['correct'])
            col3.metric("📈 Accuracy", f"{stats['accuracy']:.1%}")
            col4.metric("📅 Période", f"{stats['period_days']} jours")
            
            st.markdown("---")
            
            # Par confiance
            st.subheader("📊 Par niveau de confiance")
            
            if stats['by_confidence']:
                conf_data = []
                for label, data in stats['by_confidence'].items():
                    conf_data.append({
                        'Confiance': label,
                        'Matchs': data['count'],
                        'Correctes': data['correct'],
                        'Accuracy': f"{data['accuracy']:.1%}"
                    })
                st.dataframe(pd.DataFrame(conf_data), use_container_width=True)
            
            # Value bets
            if stats['value_bets']:
                st.subheader("💎 Value Bets")
                vb = stats['value_bets']
                c1, c2, c3 = st.columns(3)
                c1.metric("Nombre", vb['count'])
                c2.metric("Accuracy", f"{vb['accuracy']:.1%}")
                c3.metric("Edge moyen", f"{vb['avg_edge']:.1%}")
            
            # Historique récent
            st.subheader("📋 Dernières prédictions")
            recent = tracker.get_recent_predictions(days=7)
            if len(recent) > 0:
                display_cols = ['date', 'home_team', 'away_team', 'pred_winner', 'pred_confidence', 'actual_winner', 'correct']
                display = recent[[c for c in display_cols if c in recent.columns]].copy()
                display['correct'] = display['correct'].map({True: '✅', False: '❌', None: '⏳'})
                display['pred_confidence'] = display['pred_confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else '')
                st.dataframe(display, use_container_width=True)
        
        # Bouton pour mettre à jour les résultats
        st.markdown("---")
        if st.button("🔄 Mettre à jour les résultats d'hier"):
            with st.spinner("Récupération des résultats..."):
                tracker.update_results()
            st.success("✅ Résultats mis à jour!")
            st.rerun()


def save_predictions_today(tracker, model, team_stats, last_dates, games, features_list, odds_data):
    """Sauvegarde les prédictions du jour."""
    predictions = []
    
    for _, game in games.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        if home_id not in team_stats.index or away_id not in team_stats.index:
            continue
        
        home_data = team_stats.loc[home_id]
        away_data = team_stats.loc[away_id]
        home_name = home_data.get('TEAM_NAME', f'Team {home_id}')
        away_name = away_data.get('TEAM_NAME', f'Team {away_id}')
        
        home_rest = (datetime.now() - last_dates[home_id]).days
        away_rest = (datetime.now() - last_dates[away_id]).days
        
        prob_home, _ = predict_match(model, home_data.to_dict(), away_data.to_dict(), home_rest, away_rest, features_list)
        market_home, _ = match_odds(odds_data, home_name, away_name)
        
        predictions.append({
            'game_id': str(game.get('GAME_ID', '')),
            'home_team': home_name,
            'away_team': away_name,
            'pred_home_prob': prob_home,
            'market_home_prob': market_home
        })
    
    tracker.save_predictions(predictions)


if __name__ == "__main__":
    main()
