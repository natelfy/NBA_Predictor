import streamlit as st 
import pandas as pd 
import numpy as np
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import os
import sys

# Ajouter src au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import du collecteur de cotes
try:
    from src.odds_collector import OddsCollector, normalize_team_name
    ODDS_AVAILABLE = True
except ImportError:
    ODDS_AVAILABLE = False
    print("⚠️ Module odds_collector non trouvé")

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


@st.cache_data(ttl=600)
def get_today_schedule():
    """Récupère les matchs du jour."""
    try:
        today = datetime.now()
        board = scoreboardv2.ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
        return board.game_header.get_data_frame()
    except Exception as e:
        st.error(f"Erreur API NBA: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache 30 minutes
def get_odds_data():
    """Récupère les cotes des bookmakers."""
    if not ODDS_AVAILABLE:
        return None
    
    try:
        collector = OddsCollector()
        odds = collector.get_upcoming_odds()
        return odds if not odds.empty else None
    except Exception as e:
        print(f"Erreur cotes: {e}")
        return None


def match_odds_to_game(odds_df, home_name, away_name):
    """
    Trouve les cotes correspondant à un match.
    
    Returns:
        Tuple (prob_home, prob_away) ou (None, None)
    """
    if odds_df is None or odds_df.empty:
        return None, None
    
    for _, row in odds_df.iterrows():
        # Matching flexible sur les noms d'équipe
        odds_home = row['home_team'].lower()
        odds_away = row['away_team'].lower()
        
        home_match = (home_name.lower() in odds_home or 
                     odds_home in home_name.lower() or
                     home_name.split()[-1].lower() in odds_home)
        
        away_match = (away_name.lower() in odds_away or 
                     odds_away in away_name.lower() or
                     away_name.split()[-1].lower() in odds_away)
        
        if home_match and away_match:
            return row['implied_prob_home'], row['implied_prob_away']
    
    return None, None


# =============================================================================
# FONCTION DE PRÉDICTION
# =============================================================================

def predict_matchup(model, home_stats, away_stats, home_rest, away_rest, features_list):
    """
    Prédit le résultat d'un match.
    
    Returns:
        (probabilité victoire domicile, dict features utilisées)
    """
    features = {}
    
    # Features différentielles
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
    
    # Features contextuelles
    if 'HOME_IS_B2B' in features_list:
        features['HOME_IS_B2B'] = 1 if home_rest <= 1 else 0
    
    if 'AWAY_IS_B2B' in features_list:
        features['AWAY_IS_B2B'] = 1 if away_rest <= 1 else 0
    
    if 'HOME_REST_DAYS' in features_list:
        features['HOME_REST_DAYS'] = min(home_rest, 7)
    
    if 'AWAY_REST_DAYS' in features_list:
        features['AWAY_REST_DAYS'] = min(away_rest, 7)
    
    # Features absolues (fallback)
    for prefix in ['HOME_', 'AWAY_']:
        stats = home_stats if prefix == 'HOME_' else away_stats
        for col in ['ELO_PRE', 'AVG_PTS_10', 'AVG_EFG_PCT_10', 'AVG_TOV_PCT_10', 
                    'AVG_OREB_PCT_10', 'AVG_FT_RATE_10', 'AVG_NET_RTG_10']:
            feat_name = f'{prefix}{col}'
            if feat_name in features_list:
                features[feat_name] = stats.get(col, 0)
    
    # Construire le DataFrame
    X = pd.DataFrame([{f: features.get(f, 0) for f in features_list}])
    
    # Prédiction
    prob_home = model.predict_proba(X)[0][1]
    
    return prob_home, features


def calculate_value_bet(model_prob: float, market_prob: float, threshold: float = 0.05) -> dict:
    """
    Calcule si il y a une opportunité de "value bet".
    
    Un value bet existe quand notre modèle donne une probabilité
    significativement différente du marché.
    
    Args:
        model_prob: Notre probabilité
        market_prob: Probabilité du marché
        threshold: Écart minimum pour signaler (5% par défaut)
    
    Returns:
        Dict avec les informations de value
    """
    edge = model_prob - market_prob
    
    return {
        'edge': edge,
        'is_value': abs(edge) >= threshold,
        'direction': 'HOME' if edge > 0 else 'AWAY',
        'confidence': 'HIGH' if abs(edge) >= 0.10 else 'MEDIUM' if abs(edge) >= 0.05 else 'LOW'
    }


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
        odds_data = get_odds_data()
    
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
    
    # Status bar
    status_parts = [f"✅ {len(schedule)} matchs"]
    if odds_data is not None:
        status_parts.append(f"🎰 Cotes dispo")
    else:
        status_parts.append(f"⚠️ Cotes indispo")
    
    st.success(" | ".join(status_parts))
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Informations")
        
        # Métriques du modèle
        if os.path.exists('models/metrics.txt'):
            with open('models/metrics.txt', 'r') as f:
                st.text(f.read())
        
        st.markdown("---")
        
        # Configuration des cotes
        st.subheader("🎰 Configuration Cotes")
        
        if not ODDS_AVAILABLE:
            st.warning("Module cotes non installé")
            st.code("pip install requests")
        elif odds_data is None:
            st.warning("API cotes non configurée")
            st.caption("1. Crée un compte sur the-odds-api.com")
            st.caption("2. Ajoute ta clé dans src/odds_collector.py")
        else:
            st.success(f"{len(odds_data)} matchs avec cotes")
        
        st.markdown("---")
        st.caption(f"Features: {len(features_list)}")
        with st.expander("Voir les features"):
            for f in features_list:
                st.caption(f"• {f}")
    
    # --- Affichage des matchs ---
    st.markdown("---")
    
    for idx, game in schedule.iterrows():
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
        
        # Prédiction du modèle
        prob_home, used_features = predict_matchup(
            model, 
            home_data.to_dict(), 
            away_data.to_dict(),
            home_rest, 
            away_rest,
            features_list
        )
        prob_away = 1 - prob_home
        
        # Cotes du marché
        market_home, market_away = match_odds_to_game(odds_data, home_name, away_name)
        has_odds = market_home is not None
        
        # Value bet?
        value_info = None
        if has_odds:
            value_info = calculate_value_bet(prob_home, market_home)
        
        # --- Interface du match ---
        st.markdown("### " + "─" * 50)
        
        # Header avec noms d'équipes
        header_col1, header_col2, header_col3 = st.columns([2, 1, 2])
        
        with header_col1:
            st.markdown(f"### 🏃 {away_name}")
        with header_col2:
            st.markdown("<h3 style='text-align: center;'>@</h3>", unsafe_allow_html=True)
        with header_col3:
            st.markdown(f"### 🏠 {home_name}")
        
        # Ligne principale : Stats + Prédictions
        col1, col2, col3 = st.columns([2, 1.5, 2])
        
        # Équipe extérieure
        with col1:
            away_elo = int(away_data.get('ELO_PRE', 1500))
            away_form = away_data.get('FORM_5', 0.5)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("ELO", away_elo)
            m2.metric("Repos", f"{away_rest}j", delta="B2B!" if away_rest <= 1 else None, delta_color="inverse")
            m3.metric("Forme", f"{away_form:.0%}" if pd.notna(away_form) else "N/A")
        
        # Prédictions au centre
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Notre prédiction
            if prob_home > 0.5:
                winner = home_name
                confidence = prob_home
            else:
                winner = away_name
                confidence = prob_away
            
            # Couleur selon confiance
            if confidence >= 0.65:
                color = "#28a745"  # vert
            elif confidence >= 0.55:
                color = "#ffc107"  # orange
            else:
                color = "#6c757d"  # gris
            
            st.markdown(f"""
                <div style='text-align: center; background: #1a1a2e; padding: 15px; border-radius: 10px;'>
                    <p style='margin: 0; color: #888; font-size: 12px;'>🤖 NOTRE MODÈLE</p>
                    <h2 style='color: {color}; margin: 5px 0;'>{confidence:.0%}</h2>
                    <p style='margin: 0; font-size: 14px;'>→ {winner.split()[-1]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Cotes du marché si disponibles
            if has_odds:
                market_winner = home_name if market_home > market_away else away_name
                market_conf = max(market_home, market_away)
                
                st.markdown(f"""
                    <div style='text-align: center; background: #2a2a3e; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                        <p style='margin: 0; color: #888; font-size: 12px;'>🎰 MARCHÉ</p>
                        <h3 style='color: #17a2b8; margin: 5px 0;'>{market_conf:.0%}</h3>
                        <p style='margin: 0; font-size: 12px;'>→ {market_winner.split()[-1]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Alerte Value Bet
                if value_info and value_info['is_value']:
                    edge = value_info['edge']
                    if edge > 0:
                        value_team = home_name.split()[-1]
                        value_edge = edge
                    else:
                        value_team = away_name.split()[-1]
                        value_edge = -edge
                    
                    color_value = "#28a745" if value_info['confidence'] == 'HIGH' else "#ffc107"
                    
                    st.markdown(f"""
                        <div style='text-align: center; background: {color_value}22; 
                                    border: 1px solid {color_value}; padding: 8px; 
                                    border-radius: 5px; margin-top: 10px;'>
                            <p style='margin: 0; color: {color_value}; font-size: 12px;'>
                                💎 VALUE: {value_team} +{value_edge:.1%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Équipe domicile
        with col3:
            home_elo = int(home_data.get('ELO_PRE', 1500))
            home_form = home_data.get('FORM_5', 0.5)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("ELO", home_elo, delta=home_elo - away_elo)
            m2.metric("Repos", f"{home_rest}j", delta="B2B!" if home_rest <= 1 else None, delta_color="inverse")
            m3.metric("Forme", f"{home_form:.0%}" if pd.notna(home_form) else "N/A")
        
        # Détails (expander)
        with st.expander("📊 Détails de l'analyse"):
            det1, det2, det3 = st.columns(3)
            
            with det1:
                st.caption("**Features clés:**")
                for feat, val in used_features.items():
                    if 'DIFF' in feat:
                        color = "🟢" if val > 0 else "🔴" if val < 0 else "⚪"
                        st.caption(f"{color} {feat}: {val:+.2f}")
            
            with det2:
                st.caption("**Comparaison probas:**")
                st.caption(f"🤖 Modèle Home: {prob_home:.1%}")
                st.caption(f"🤖 Modèle Away: {prob_away:.1%}")
                if has_odds:
                    st.caption(f"🎰 Marché Home: {market_home:.1%}")
                    st.caption(f"🎰 Marché Away: {market_away:.1%}")
            
            with det3:
                st.caption("**Interprétation:**")
                elo_diff = used_features.get('ELO_PRE_DIFF', 0)
                if elo_diff > 50:
                    st.caption("✅ Domicile favori ELO")
                elif elo_diff < -50:
                    st.caption("✅ Extérieur favori ELO")
                else:
                    st.caption("⚖️ Match équilibré")
                
                if has_odds and value_info and value_info['is_value']:
                    st.caption(f"💎 Edge détecté: {abs(value_info['edge']):.1%}")
    
    # --- Footer ---
    st.markdown("---")
    
    # Légende
    leg1, leg2, leg3 = st.columns(3)
    with leg1:
        st.caption("🤖 = Prédiction de notre modèle")
    with leg2:
        st.caption("🎰 = Probabilité implicite des cotes")
    with leg3:
        st.caption("💎 = Value bet potentiel (edge >5%)")
    
    st.caption("⚠️ Ce système est à but éducatif. Les paris sportifs comportent des risques.")


if __name__ == "__main__":
    main()