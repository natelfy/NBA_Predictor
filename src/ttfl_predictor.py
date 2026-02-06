"""
NBA Oracle - TTFL Predictor
============================

Prédit le score TTFL attendu pour chaque joueur et génère des recommandations.

Facteurs pris en compte:
1. Moyenne TTFL du joueur
2. Forme récente (5 derniers matchs)
3. Historique vs adversaire
4. Home/Away
5. Back-to-Back
6. Minutes moyennes
7. Variance (risque)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.ttfl_collector import (
    TTFLStatsCollector, 
    TTFLDeckManager,
    calculate_ttfl_score
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Poids pour le score prédit
WEIGHTS = {
    'season_avg': 0.40,      # Moyenne saison
    'recent_form': 0.35,     # 5 derniers matchs
    'vs_opponent': 0.15,     # Historique vs adversaire
    'context': 0.10,         # Home/Away, B2B
}

# Bonus/Malus contextuels
CONTEXT_ADJUSTMENTS = {
    'home': 1.5,             # Bonus domicile
    'away': -1.0,            # Malus extérieur
    'b2b': -3.0,             # Malus back-to-back
    'rest_3plus': 1.0,       # Bonus repos 3+ jours
}

# Seuils de risque (écart-type)
RISK_THRESHOLDS = {
    'low': 8,                # Consistant
    'medium': 12,            # Normal
    'high': 16,              # Variable
}


# =============================================================================
# PRÉDICTEUR TTFL
# =============================================================================

class TTFLPredictor:
    """
    Prédit le score TTFL attendu pour chaque joueur.
    """
    
    def __init__(self):
        self.collector = TTFLStatsCollector()
        self.deck_manager = TTFLDeckManager()
    
    def predict_player_score(
        self, 
        player_data: pd.Series,
        recent_games: pd.DataFrame = None,
        vs_opponent_scores: List[float] = None,
        is_home: bool = True,
        is_b2b: bool = False,
        rest_days: int = 1
    ) -> Dict:
        """
        Prédit le score TTFL d'un joueur pour ce soir.
        
        Args:
            player_data: Stats moyennes du joueur
            recent_games: DataFrame des derniers matchs
            vs_opponent_scores: Scores TTFL vs cet adversaire
            is_home: Joue à domicile?
            is_b2b: Back-to-back?
            rest_days: Jours de repos
        
        Returns:
            Dict avec score prédit, intervalle de confiance, risque
        """
        # 1. Moyenne saison
        season_avg = player_data.get('TTFL_AVG', 25)
        
        # 2. Forme récente
        if recent_games is not None and len(recent_games) > 0:
            if 'TTFL_SCORE' in recent_games.columns:
                recent_scores = recent_games['TTFL_SCORE'].tolist()
            else:
                recent_scores = recent_games.apply(calculate_ttfl_score, axis=1).tolist()
            
            recent_avg = np.mean(recent_scores[:5]) if recent_scores else season_avg
            recent_std = np.std(recent_scores[:10]) if len(recent_scores) >= 3 else 10
        else:
            recent_avg = season_avg
            recent_std = 10
        
        # 3. Historique vs adversaire
        if vs_opponent_scores and len(vs_opponent_scores) > 0:
            vs_avg = np.mean(vs_opponent_scores)
        else:
            vs_avg = season_avg  # Pas d'historique = moyenne
        
        # 4. Ajustements contextuels
        context_adj = 0
        if is_home:
            context_adj += CONTEXT_ADJUSTMENTS['home']
        else:
            context_adj += CONTEXT_ADJUSTMENTS['away']
        
        if is_b2b:
            context_adj += CONTEXT_ADJUSTMENTS['b2b']
        elif rest_days >= 3:
            context_adj += CONTEXT_ADJUSTMENTS['rest_3plus']
        
        # 5. Score final pondéré
        base_score = (
            WEIGHTS['season_avg'] * season_avg +
            WEIGHTS['recent_form'] * recent_avg +
            WEIGHTS['vs_opponent'] * vs_avg +
            WEIGHTS['context'] * (season_avg + context_adj)
        )
        
        predicted_score = base_score + context_adj * 0.5
        
        # 6. Intervalle de confiance
        confidence_low = predicted_score - 1.5 * recent_std
        confidence_high = predicted_score + 1.5 * recent_std
        
        # 7. Niveau de risque
        if recent_std <= RISK_THRESHOLDS['low']:
            risk = 'low'
            risk_emoji = '🟢'
        elif recent_std <= RISK_THRESHOLDS['medium']:
            risk = 'medium'
            risk_emoji = '🟡'
        else:
            risk = 'high'
            risk_emoji = '🔴'
        
        return {
            'predicted_score': round(predicted_score, 1),
            'confidence_low': round(confidence_low, 1),
            'confidence_high': round(confidence_high, 1),
            'season_avg': round(season_avg, 1),
            'recent_avg': round(recent_avg, 1),
            'vs_opponent_avg': round(vs_avg, 1) if vs_opponent_scores else None,
            'std_dev': round(recent_std, 1),
            'risk': risk,
            'risk_emoji': risk_emoji,
            'context_adjustment': round(context_adj, 1),
            'is_home': is_home,
            'is_b2b': is_b2b,
        }
    
    def get_tonight_recommendations(
        self, 
        top_n: int = 20,
        exclude_used: bool = True,
        min_minutes: float = 20
    ) -> pd.DataFrame:
        """
        Génère les recommandations pour ce soir.
        
        Args:
            top_n: Nombre de joueurs à retourner
            exclude_used: Exclure les joueurs en cooldown
            min_minutes: Minutes minimum par match
        
        Returns:
            DataFrame avec les recommandations triées
        """
        print("🎯 Génération des recommandations TTFL...")
        
        # Joueurs du soir
        today_players = self.collector.get_today_players()
        
        if today_players.empty:
            print("   ⚠️ Pas de match ce soir")
            return pd.DataFrame()
        
        # Filtrer par minutes
        today_players = today_players[today_players['MIN'] >= min_minutes]
        
        recommendations = []
        
        for _, player in today_players.iterrows():
            player_id = player['PLAYER_ID']
            player_name = player['PLAYER_NAME']
            
            # Vérifier disponibilité
            is_available, last_pick_date = self.deck_manager.is_available(player_id)
            
            if exclude_used and not is_available:
                continue
            
            # Récupérer derniers matchs
            try:
                recent = self.collector.get_player_recent_games(player_id, n_games=10)
            except:
                recent = pd.DataFrame()
            
            # Historique vs adversaire
            opponent_id = player.get('OPPONENT_ID')
            if opponent_id:
                try:
                    vs_history = self.collector.get_player_vs_team_history(
                        player_id, int(opponent_id), n_games=5
                    )
                except:
                    vs_history = []
            else:
                vs_history = []
            
            # Contexte
            is_home = player.get('HOME_AWAY') == 'HOME'
            # TODO: détecter B2B (nécessite l'historique des matchs de l'équipe)
            is_b2b = False
            
            # Prédiction
            pred = self.predict_player_score(
                player,
                recent_games=recent,
                vs_opponent_scores=vs_history,
                is_home=is_home,
                is_b2b=is_b2b
            )
            
            recommendations.append({
                'player_id': player_id,
                'player_name': player_name,
                'team': player.get('TEAM_ABBREVIATION', ''),
                'opponent_id': opponent_id,
                'home_away': '🏠' if is_home else '✈️',
                'predicted_score': pred['predicted_score'],
                'confidence_low': pred['confidence_low'],
                'confidence_high': pred['confidence_high'],
                'season_avg': pred['season_avg'],
                'recent_avg': pred['recent_avg'],
                'vs_opponent': pred['vs_opponent_avg'],
                'risk': pred['risk'],
                'risk_emoji': pred['risk_emoji'],
                'std_dev': pred['std_dev'],
                'minutes': player.get('MIN', 0),
                'available': is_available,
                'available_date': self.deck_manager.get_available_date(player_id) if not is_available else None,
            })
        
        # Créer DataFrame
        df = pd.DataFrame(recommendations)
        
        if df.empty:
            return df
        
        # Trier par score prédit
        df = df.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        
        # Ajouter rang
        df['rank'] = range(1, len(df) + 1)
        
        print(f"   ✅ {len(df)} joueurs analysés")
        
        return df.head(top_n)
    
    def get_value_picks(self, recommendations: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Identifie les "value picks" - joueurs sous-côtés avec bon potentiel.
        
        Critères:
        - Score prédit > 35
        - Risque low ou medium
        - Pas dans le top 10 des plus connus
        """
        if recommendations.empty:
            return pd.DataFrame()
        
        # Filtrer les critères de value
        value = recommendations[
            (recommendations['predicted_score'] >= 30) &
            (recommendations['risk'].isin(['low', 'medium'])) &
            (recommendations['rank'] > 10)  # Pas les ultra-stars
        ].copy()
        
        # Score de value = predicted / season_avg (surperformance attendue)
        value['value_score'] = value['predicted_score'] / value['season_avg']
        
        return value.sort_values('value_score', ascending=False).head(top_n)
    
    def get_safe_picks(self, recommendations: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Identifie les picks "safe" - consistants avec faible variance.
        
        Critères:
        - Risque low
        - Minutes > 30
        - Score prédit > 30
        """
        if recommendations.empty:
            return pd.DataFrame()
        
        safe = recommendations[
            (recommendations['risk'] == 'low') &
            (recommendations['minutes'] >= 30) &
            (recommendations['predicted_score'] >= 30)
        ].copy()
        
        # Score de sûreté = predicted / std_dev
        safe['safety_score'] = safe['predicted_score'] / (safe['std_dev'] + 1)
        
        return safe.sort_values('safety_score', ascending=False).head(top_n)
    
    def get_high_ceiling_picks(self, recommendations: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Identifie les picks "high ceiling" - potentiel d'explosion.
        
        Critères:
        - Confidence_high > 60
        - Capable de gros matchs
        """
        if recommendations.empty:
            return pd.DataFrame()
        
        ceiling = recommendations[
            recommendations['confidence_high'] >= 55
        ].copy()
        
        return ceiling.sort_values('confidence_high', ascending=False).head(top_n)


# =============================================================================
# STRATÉGIES DE PICK
# =============================================================================

def suggest_strategy(deck_manager: TTFLDeckManager, days_in_season: int = 180) -> str:
    """
    Suggère une stratégie en fonction de l'état du deck.
    
    Args:
        deck_manager: Gestionnaire de deck
        days_in_season: Jours estimés dans la saison
    
    Returns:
        Recommandation stratégique
    """
    stats = deck_manager.get_stats()
    picks_used = stats['players_used_season']
    
    # Estimation: ~450 joueurs, ~180 jours de saison
    # On peut utiliser chaque joueur 6 fois max (180/30)
    
    if picks_used < 30:
        return """
🎯 **STRATÉGIE: DÉBUT DE SAISON**

Tu as encore beaucoup de stars disponibles!

Recommandations:
• Utilise tes TOP picks quand ils ont des matchups faciles
• Garde 2-3 superstars en réserve pour les moments clés
• N'hésite pas à prendre des risques sur les gros scores
        """
    
    elif picks_used < 80:
        return """
🎯 **STRATÉGIE: MI-SAISON**

Tu as utilisé une bonne partie de ton deck.

Recommandations:
• Commence à chercher des value picks (joueurs moins connus)
• Utilise les stars restantes avec parcimonie
• Surveille les opportunités (star OUT = backup explose)
        """
    
    else:
        return """
🎯 **STRATÉGIE: FIN DE SAISON**

Ton deck est bien entamé!

Recommandations:
• Privilégie les picks "safe" pour maintenir ton score
• Cherche les joueurs que tu n'as jamais utilisés
• Les role players avec gros temps de jeu deviennent précieux
        """


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 TEST TTFL PREDICTOR")
    print("=" * 60)
    
    predictor = TTFLPredictor()
    
    # Recommandations du soir
    recs = predictor.get_tonight_recommendations(top_n=15)
    
    if not recs.empty:
        print(f"\n📊 TOP 15 PICKS CE SOIR:")
        print("-" * 70)
        print(f"{'#':>2} {'Joueur':<22} {'Team':<4} {'Loc':<3} {'Prédit':>6} {'Risque':<6}")
        print("-" * 70)
        
        for _, row in recs.iterrows():
            print(f"{row['rank']:2d}. {row['player_name'][:20]:<22} "
                  f"{row['team']:<4} {row['home_away']:<3} "
                  f"{row['predicted_score']:>5.1f} {row['risk_emoji']}")
        
        # Value picks
        print(f"\n💎 VALUE PICKS:")
        value = predictor.get_value_picks(recs)
        for _, row in value.iterrows():
            print(f"   • {row['player_name']} ({row['predicted_score']:.1f} pts)")
        
        # Safe picks
        print(f"\n🛡️ SAFE PICKS:")
        safe = predictor.get_safe_picks(recs)
        for _, row in safe.iterrows():
            print(f"   • {row['player_name']} ({row['predicted_score']:.1f} pts, σ={row['std_dev']:.1f})")
