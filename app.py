"""
NBA Oracle - Version Finale
============================

Interface Streamlit avec:
- 🏥 Blessures en temps réel (Balldontlie API)
- 🎰 Cotes en temps réel
- 📊 Prédictions ML ajustées
- 📈 Tracking des performances

API: Balldontlie (gratuit, pas besoin de Java!)
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

# =============================================================================
# IMPORTS OPTIONNELS
# =============================================================================

# Balldontlie (nouvelle API unifiée)
try:
    from src.balldontlie_collector import BalldontlieCollector
    BALLDONTLIE_AVAILABLE = True
except ImportError:
    BALLDONTLIE_AVAILABLE = False

# Tracker
try:
    from src.tracker import PredictionTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

# NBA API (fallback pour les matchs)
try:
    from nba_api.stats.endpoints import scoreboardv2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# Ancien collecteur de cotes (fallback)
try:
    from src.odds_collector import OddsCollector
    ODDS_COLLECTOR_AVAILABLE = True
except ImportError:
    ODDS_COLLECTOR_AVAILABLE = False

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

# Récupérer la clé API depuis les secrets Streamlit ou env
BALLDONTLIE_API_KEY = st.secrets.get("BALLDONTLIE_API_KEY", os.environ.get("BALLDONTLIE_API_KEY", ""))


# =============================================================================
# CHARGEMENT (avec cache)
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
def get_today_games_nba_api():
    """Récupère les matchs via NBA API (fallback)."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    try:
        board = scoreboardv2.ScoreboardV2(
            game_date=datetime.now().strftime('%Y-%m-%d'), 
            timeout=30
        )
        return board.game_header.get_data_frame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_today_games_balldontlie():
    """Récupère les matchs via Balldontlie."""
    if not BALLDONTLIE_AVAILABLE:
        return pd.DataFrame()
    try:
        collector = BalldontlieCollector(api_key=BALLDONTLIE_API_KEY)
        return collector.get_today_games()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_injuries():
    """Récupère les blessures via Balldontlie."""
    if not BALLDONTLIE_AVAILABLE:
        return pd.DataFrame()
    try:
        collector = BalldontlieCollector(api_key=BALLDONTLIE_API_KEY)
        return collector.get_injuries()
    except Exception as e:
        print(f"Erreur blessures: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_odds_balldontlie():
    """Récupère les cotes via Balldontlie."""
    if not BALLDONTLIE_AVAILABLE:
        return pd.DataFrame()
    try:
        collector = BalldontlieCollector(api_key=BALLDONTLIE_API_KEY)
        return collector.get_today_odds()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_odds_fallback():
    """Fallback: cotes via The Odds API."""
    if not ODDS_COLLECTOR_AVAILABLE:
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

def predict_match(model, home_stats, away_stats, home_rest, away_rest, features_list, injury_adj=0.0):
    """
    Prédit le résultat d'un match.
    
    Args:
        injury_adj: Ajustement basé sur les blessures (-0.1 à +0.1)
    """
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
    base_prob = model.predict_proba(X)[0][1]
    
    # Appliquer l'ajustement blessures
    adjusted_prob = np.clip(base_prob + injury_adj, 0.01, 0.99)
    
    return adjusted_prob, features

def get_injury_info(home_name, away_name, injuries_df):
    """Récupère les infos de blessures pour un matchup."""
    if injuries_df is None or injuries_df.empty or not BALLDONTLIE_AVAILABLE:
        return 0.0, None, None
    
    try:
        collector = BalldontlieCollector(api_key=BALLDONTLIE_API_KEY)
        matchup = collector.get_matchup_injury_diff(home_name, away_name, injuries_df)
        return (
            matchup['differential'],
            matchup['home_impact'],
            matchup['away_impact']
        )
    except:
        return 0.0, None, None

def match_odds(home_name, away_name, odds_balldontlie, odds_fallback):
    """Trouve les cotes pour un matchup."""
    # Essayer Balldontlie d'abord
    if odds_balldontlie is not None and not odds_balldontlie.empty:
        if BALLDONTLIE_AVAILABLE:
            collector = BalldontlieCollector(api_key=BALLDONTLIE_API_KEY)
            prob_h, prob_a = collector.match_odds(home_name, away_name, odds_balldontlie)
            if prob_h is not None:
                return prob_h, prob_a
    
    # Fallback sur The Odds API
    if odds_fallback is not None:
        for _, row in odds_fallback.iterrows():
            h = home_name.lower() in row['home_team'].lower() or home_name.split()[-1].lower() in row['home_team'].lower()
            a = away_name.lower() in row['away_team'].lower() or away_name.split()[-1].lower() in row['away_team'].lower()
            if h and a:
                return row.get('implied_prob_home'), row.get('implied_prob_away')
    
    return None, None


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    # Header
    st.title("🏀 NBA Oracle")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Prédictions du jour", "📈 Performance (Tracking)"])
    
    # Chargement des données
    model = load_model()
    team_stats, last_dates = load_team_stats()
    features_list = load_features()
    
    # Matchs (préférer NBA API pour les IDs d'équipe compatibles)
    games = get_today_games_nba_api()
    if games.empty:
        games_bdl = get_today_games_balldontlie()
    
    # Blessures (Balldontlie)
    injuries_data = get_injuries()
    
    # Cotes (Balldontlie + fallback)
    odds_bdl = get_odds_balldontlie()
    odds_fallback = get_odds_fallback()
    
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
        
        # Status bar
        status_items = [f"✅ {len(games)} match(s)"]
        
        if not injuries_data.empty:
            status_items.append(f"🏥 {len(injuries_data)} blessés")
        else:
            status_items.append("⚠️ Blessures N/A")
        
        if (odds_bdl is not None and not odds_bdl.empty) or odds_fallback is not None:
            status_items.append("🎰 Cotes dispo")
        else:
            status_items.append("⚠️ Cotes N/A")
        
        col_info, col_save = st.columns([3, 1])
        with col_info:
            st.success(" | ".join(status_items))
        with col_save:
            if tracker and st.button("💾 Sauvegarder"):
                save_predictions_today(
                    tracker, model, team_stats, last_dates, games, 
                    features_list, injuries_data, odds_bdl, odds_fallback
                )
                st.success("✅ Sauvegardé!")
        
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
            
            # Blessures
            injury_adj, home_injuries, away_injuries = get_injury_info(
                home_name, away_name, injuries_data
            )
            
            # Prédiction (avec ajustement blessures)
            prob_home, _ = predict_match(
                model, home_data.to_dict(), away_data.to_dict(), 
                home_rest, away_rest, features_list, injury_adj
            )
            
            # Cotes
            market_home, _ = match_odds(home_name, away_name, odds_bdl, odds_fallback)
            
            # ===== AFFICHAGE =====
            col1, col2, col3 = st.columns([2, 1.5, 2])
            
            # Away team
            with col1:
                st.markdown(f"### 🏃 {away_name}")
                away_elo = int(away_data.get('ELO_PRE', 1500))
                
                # Info ligne
                info_parts = [f"ELO: **{away_elo}**"]
                if away_rest <= 1:
                    info_parts.append("⚡ B2B")
                st.markdown(" | ".join(info_parts))
                
                # Blessures
                if away_injuries and away_injuries['out_players']:
                    with st.expander(f"🏥 {len(away_injuries['out_players'])} absent(s)", expanded=False):
                        for p in away_injuries['out_players'][:5]:
                            emoji = "⭐" if p['tier'] in ['superstar', 'allstar'] else "•"
                            st.markdown(f"{emoji} **{p['name']}** - {p['status']}")
            
            # Prédiction centrale
            with col2:
                winner = home_name if prob_home > 0.5 else away_name
                conf = max(prob_home, 1 - prob_home)
                
                if conf >= 0.65:
                    color = "#28a745"  # Vert
                elif conf >= 0.55:
                    color = "#ffc107"  # Jaune
                else:
                    color = "#6c757d"  # Gris
                
                st.markdown(f"""
                    <div style='text-align: center; background: #1a1a2e; padding: 15px; border-radius: 10px; margin-top: 10px;'>
                        <p style='margin: 0; color: #888; font-size: 0.9em;'>🤖 PRÉDICTION</p>
                        <h2 style='color: {color}; margin: 5px 0; font-size: 2.5em;'>{conf:.0%}</h2>
                        <p style='margin: 0;'>→ <strong>{winner.split()[-1]}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Indicateurs additionnels
                extras = []
                
                # Impact blessures
                if abs(injury_adj) >= 0.015:
                    if injury_adj > 0:
                        extras.append(f"🏥 +{injury_adj:.1%} home")
                    else:
                        extras.append(f"🏥 {injury_adj:.1%} away")
                
                # Value bet
                if market_home:
                    edge = prob_home - market_home
                    if abs(edge) >= 0.05:
                        vt = home_name.split()[-1] if edge > 0 else away_name.split()[-1]
                        extras.append(f"💎 VALUE {vt} +{abs(edge):.1%}")
                
                if extras:
                    st.markdown(f"<div style='text-align:center; margin-top: 10px; color: #ffc107;'>{'<br>'.join(extras)}</div>", unsafe_allow_html=True)
            
            # Home team
            with col3:
                st.markdown(f"### 🏠 {home_name}")
                home_elo = int(home_data.get('ELO_PRE', 1500))
                
                info_parts = [f"ELO: **{home_elo}**"]
                if home_rest <= 1:
                    info_parts.append("⚡ B2B")
                st.markdown(" | ".join(info_parts))
                
                # Blessures
                if home_injuries and home_injuries['out_players']:
                    with st.expander(f"🏥 {len(home_injuries['out_players'])} absent(s)", expanded=False):
                        for p in home_injuries['out_players'][:5]:
                            emoji = "⭐" if p['tier'] in ['superstar', 'allstar'] else "•"
                            st.markdown(f"{emoji} **{p['name']}** - {p['status']}")
            
            st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>
            🏥 Blessures temps réel | 🎰 Cotes live | 🔄 MAJ auto quotidienne<br>
            <em>Powered by Balldontlie API</em>
        </div>
        """, unsafe_allow_html=True)
    
    # =================================
    # TAB 2: TRACKING
    # =================================
    with tab2:
        st.header("📈 Performance du modèle")
        
        if not TRACKER_AVAILABLE or tracker is None:
            st.warning("⚠️ Module tracker non disponible")
            st.stop()
        
        stats = tracker.get_stats(days=30)
        
        if stats['total_predictions'] == 0:
            st.info("📭 Aucune prédiction trackée pour l'instant.")
            st.markdown("""
            **Comment ça marche :**
            1. Clique sur **💾 Sauvegarder** dans l'onglet Prédictions
            2. Le lendemain, les résultats sont automatiquement récupérés
            3. Reviens ici voir ta vraie performance !
            """)
            
            if 'pending' in stats:
                st.info(f"⏳ {stats['pending']} prédictions en attente")
        else:
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Prédictions", stats['total_predictions'])
            col2.metric("✅ Correctes", stats['correct'])
            col3.metric("📈 Accuracy", f"{stats['accuracy']:.1%}")
            col4.metric("📅 Période", f"{stats['period_days']}j")
            
            st.markdown("---")
            
            # Par confiance
            if stats['by_confidence']:
                st.subheader("📊 Accuracy par confiance")
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
                st.subheader("💎 Performance Value Bets")
                vb = stats['value_bets']
                c1, c2, c3 = st.columns(3)
                c1.metric("Nombre", vb['count'])
                c2.metric("Accuracy", f"{vb['accuracy']:.1%}")
                c3.metric("Edge moyen", f"{vb['avg_edge']:.1%}")
            
            # Historique
            st.subheader("📋 Dernières prédictions")
            recent = tracker.get_recent_predictions(days=7)
            if len(recent) > 0:
                cols = ['date', 'home_team', 'away_team', 'pred_winner', 'pred_confidence', 'actual_winner', 'correct']
                display = recent[[c for c in cols if c in recent.columns]].copy()
                display['correct'] = display['correct'].map({True: '✅', False: '❌', None: '⏳'})
                display['pred_confidence'] = display['pred_confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else '')
                st.dataframe(display, use_container_width=True)
        
        st.markdown("---")
        if st.button("🔄 MAJ résultats d'hier"):
            with st.spinner("Récupération..."):
                tracker.update_results()
            st.success("✅ Mis à jour!")
            st.rerun()


def save_predictions_today(tracker, model, team_stats, last_dates, games, features_list, injuries_data, odds_bdl, odds_fallback):
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
        
        injury_adj, _, _ = get_injury_info(home_name, away_name, injuries_data)
        
        prob_home, _ = predict_match(
            model, home_data.to_dict(), away_data.to_dict(), 
            home_rest, away_rest, features_list, injury_adj
        )
        
        market_home, _ = match_odds(home_name, away_name, odds_bdl, odds_fallback)
        
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
