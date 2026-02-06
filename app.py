"""
NBA Oracle - Application avec TTFL
===================================

Interface complète avec:
- Prédictions des matchs NBA
- Tracking des prédictions
- Module TTFL (picks quotidiens)
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
    from src.ttfl_predictor import TTFLPredictor, suggest_strategy
    from src.ttfl_collector import TTFLDeckManager
    TTFL_AVAILABLE = True
except ImportError:
    TTFL_AVAILABLE = False

try:
    from nba_api.stats.endpoints import scoreboardv2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="🏀 NBA Oracle + TTFL",
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
# PRÉDICTION MATCHS
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
    st.title("🏀 NBA Oracle + TTFL")
    
    # Tabs
    if TTFL_AVAILABLE:
        tab1, tab2, tab3 = st.tabs(["📊 Prédictions Matchs", "🎯 TTFL Picks", "📈 Tracking"])
    else:
        tab1, tab2 = st.tabs(["📊 Prédictions Matchs", "📈 Tracking"])
        tab3 = None
    
    # Chargement
    model = load_model()
    team_stats, last_dates = load_team_stats()
    features_list = load_features()
    games = get_today_games()
    odds_data = get_odds()
    
    # Tracker
    tracker = PredictionTracker() if TRACKER_AVAILABLE else None
    
    # =================================
    # TAB 1: PRÉDICTIONS MATCHS
    # =================================
    with tab1:
        st.markdown(f"**{datetime.now().strftime('%A %d %B %Y')}**")
        
        if model is None:
            st.error("❌ Modèle non trouvé")
        elif team_stats is None:
            st.error("❌ Données non trouvées")
        elif games.empty:
            st.warning("📅 Aucun match aujourd'hui")
        else:
            # Bouton sauvegarder
            col_info, col_save = st.columns([3, 1])
            with col_info:
                st.success(f"✅ {len(games)} match(s) | {'🎰 Cotes dispo' if odds_data is not None else '⚠️ Pas de cotes'}")
            with col_save:
                if tracker and st.button("💾 Sauvegarder prédictions"):
                    save_predictions_today(tracker, model, team_stats, last_dates, games, features_list, odds_data)
                    st.success("✅ Sauvegardé!")
            
            st.markdown("---")
            
            # Matchs
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
                
                col1, col2, col3 = st.columns([2, 1.5, 2])
                
                with col1:
                    st.markdown(f"### 🏃 {away_name}")
                    st.metric("ELO", int(away_data.get('ELO_PRE', 1500)))
                
                with col2:
                    winner = home_name if prob_home > 0.5 else away_name
                    conf = max(prob_home, 1 - prob_home)
                    color = "#28a745" if conf >= 0.65 else "#ffc107" if conf >= 0.55 else "#6c757d"
                    
                    st.markdown(f"""
                        <div style='text-align: center; background: #1a1a2e; padding: 15px; border-radius: 10px;'>
                            <p style='margin: 0; color: #888;'>🤖 PRÉDICTION</p>
                            <h2 style='color: {color}; margin: 5px 0;'>{conf:.0%}</h2>
                            <p>→ {winner.split()[-1]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"### 🏠 {home_name}")
                    st.metric("ELO", int(home_data.get('ELO_PRE', 1500)))
                
                st.markdown("---")
    
    # =================================
    # TAB 2 ou 3: TTFL
    # =================================
    if TTFL_AVAILABLE and tab3:
        with tab2:
            render_ttfl_tab()
    
    # =================================
    # TAB TRACKING
    # =================================
    tracking_tab = tab3 if TTFL_AVAILABLE else tab2
    with tracking_tab:
        render_tracking_tab(tracker)


def render_ttfl_tab():
    """Affiche l'onglet TTFL."""
    
    st.header("🎯 TTFL - Recommandations du soir")
    
    # Initialiser le predictor
    predictor = TTFLPredictor()
    deck_manager = TTFLDeckManager()
    
    # Sidebar: Stats du deck
    with st.sidebar:
        st.markdown("### 📊 Mon Deck TTFL")
        stats = deck_manager.get_stats()
        st.metric("Joueurs utilisés (30j)", stats['picks_30_days'])
        st.metric("Joueurs bloqués", stats['players_locked'])
        st.metric("Total saison", stats['players_used_season'])
        
        # Stratégie
        strategy = suggest_strategy(deck_manager)
        st.markdown(strategy)
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        exclude_used = st.checkbox("Exclure joueurs en cooldown", value=True)
    with col2:
        min_minutes = st.slider("Minutes minimum", 15, 35, 20)
    with col3:
        top_n = st.slider("Nombre de joueurs", 10, 30, 15)
    
    # Bouton de génération
    if st.button("🔄 Générer les recommandations", type="primary"):
        with st.spinner("Analyse en cours..."):
            recs = predictor.get_tonight_recommendations(
                top_n=top_n,
                exclude_used=exclude_used,
                min_minutes=min_minutes
            )
        
        if recs.empty:
            st.warning("⚠️ Pas de match ce soir ou données indisponibles")
        else:
            st.session_state['ttfl_recs'] = recs
    
    # Afficher les recommandations
    if 'ttfl_recs' in st.session_state and not st.session_state['ttfl_recs'].empty:
        recs = st.session_state['ttfl_recs']
        
        st.markdown("---")
        
        # TOP PICKS
        st.subheader("🏆 TOP PICKS CE SOIR")
        
        # Tableau principal
        display_cols = ['rank', 'player_name', 'team', 'home_away', 
                       'predicted_score', 'season_avg', 'risk_emoji', 'minutes']
        
        display_df = recs[display_cols].copy()
        display_df.columns = ['#', 'Joueur', 'Team', 'Loc', 'Prédit', 'Moy.', 'Risque', 'Min']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '#': st.column_config.NumberColumn(width="small"),
                'Joueur': st.column_config.TextColumn(width="medium"),
                'Team': st.column_config.TextColumn(width="small"),
                'Loc': st.column_config.TextColumn(width="small"),
                'Prédit': st.column_config.NumberColumn(format="%.1f"),
                'Moy.': st.column_config.NumberColumn(format="%.1f"),
                'Risque': st.column_config.TextColumn(width="small"),
                'Min': st.column_config.NumberColumn(format="%.0f"),
            }
        )
        
        # Analyses spécifiques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🛡️ SAFE PICKS")
            st.caption("Consistants, faible variance")
            safe = predictor.get_safe_picks(recs, top_n=3)
            for _, row in safe.iterrows():
                st.markdown(f"**{row['player_name']}** ({row['team']})")
                st.caption(f"Prédit: {row['predicted_score']:.1f} | σ={row['std_dev']:.1f}")
        
        with col2:
            st.markdown("### 💎 VALUE PICKS")
            st.caption("Sous-cotés, bon potentiel")
            value = predictor.get_value_picks(recs, top_n=3)
            for _, row in value.iterrows():
                st.markdown(f"**{row['player_name']}** ({row['team']})")
                st.caption(f"Prédit: {row['predicted_score']:.1f} | Moy: {row['season_avg']:.1f}")
        
        with col3:
            st.markdown("### 🚀 HIGH CEILING")
            st.caption("Potentiel d'explosion")
            ceiling = predictor.get_high_ceiling_picks(recs, top_n=3)
            for _, row in ceiling.iterrows():
                st.markdown(f"**{row['player_name']}** ({row['team']})")
                st.caption(f"Ceiling: {row['confidence_high']:.1f}")
        
        st.markdown("---")
        
        # Enregistrer un pick
        st.subheader("📝 Enregistrer mon pick")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_player = st.selectbox(
                "Joueur sélectionné ce soir",
                options=recs['player_name'].tolist(),
                index=0
            )
        
        with col2:
            if st.button("✅ Valider le pick"):
                player_row = recs[recs['player_name'] == selected_player].iloc[0]
                deck_manager.add_pick(
                    player_id=int(player_row['player_id']),
                    player_name=selected_player
                )
                st.success(f"✅ {selected_player} ajouté à ton deck!")
                st.balloons()
        
        # Historique des picks récents
        st.markdown("---")
        st.subheader("📋 Mes picks récents")
        
        recent_picks = deck_manager.get_recent_picks(30)
        if recent_picks:
            picks_df = pd.DataFrame(recent_picks)
            picks_df = picks_df[['date', 'player_name', 'days_ago', 'available_in']]
            picks_df.columns = ['Date', 'Joueur', 'Il y a (j)', 'Dispo dans (j)']
            st.dataframe(picks_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun pick enregistré ces 30 derniers jours")


def render_tracking_tab(tracker):
    """Affiche l'onglet tracking."""
    
    st.header("📈 Performance du modèle")
    
    if not TRACKER_AVAILABLE or tracker is None:
        st.warning("⚠️ Module tracker non disponible")
        return
    
    stats = tracker.get_stats(days=30)
    
    if stats['total_predictions'] == 0:
        st.info("📭 Aucune prédiction trackée pour l'instant.")
        st.markdown("""
        **Comment ça marche :**
        1. Va dans l'onglet "Prédictions Matchs"
        2. Clique sur "💾 Sauvegarder prédictions"
        3. Le lendemain, les résultats seront récupérés automatiquement
        """)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Prédictions", stats['total_predictions'])
        col2.metric("✅ Correctes", stats['correct'])
        col3.metric("📈 Accuracy", f"{stats['accuracy']:.1%}")
        col4.metric("📅 Période", f"{stats['period_days']} jours")
        
        if stats['by_confidence']:
            st.subheader("Par niveau de confiance")
            conf_data = [{'Confiance': k, 'Matchs': v['count'], 'Accuracy': f"{v['accuracy']:.1%}"} 
                        for k, v in stats['by_confidence'].items()]
            st.dataframe(pd.DataFrame(conf_data), use_container_width=True)
    
    if st.button("🔄 Mettre à jour les résultats d'hier"):
        with st.spinner("Récupération..."):
            tracker.update_results()
        st.success("✅ Mis à jour!")
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
