"""
NBA Oracle - Balldontlie API Integration
=========================================

Module unifié pour récupérer:
- 🏥 Blessures du jour
- 🎰 Cotes en temps réel
- 📊 Stats et matchs

Utilise l'API Balldontlie (tier gratuit: 5 req/min)

Installation:
    pip install balldontlie

Usage:
    from src.balldontlie_collector import BalldontlieCollector
    
    collector = BalldontlieCollector(api_key="YOUR_KEY")
    injuries = collector.get_today_injuries()
    odds = collector.get_today_odds()
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Impact estimé par position/tier de joueur
PLAYER_IMPACT = {
    'superstar': 0.04,    # LeBron, Giannis, etc.
    'allstar': 0.025,     # All-Stars
    'starter': 0.015,     # Titulaires
    'rotation': 0.008,    # Rotation
    'bench': 0.003,       # Banc
}

# Superstars connues (à jour 2025-26)
SUPERSTARS = [
    'LeBron James', 'Giannis Antetokounmpo', 'Luka Doncic', 'Nikola Jokic',
    'Stephen Curry', 'Kevin Durant', 'Jayson Tatum', 'Joel Embiid',
    'Ja Morant', 'Anthony Edwards', 'Devin Booker', 'Donovan Mitchell',
    'Shai Gilgeous-Alexander', 'Jaylen Brown', 'Anthony Davis',
    'Kawhi Leonard', 'Damian Lillard', 'Victor Wembanyama',
    'Tyrese Haliburton', 'Paolo Banchero', 'Trae Young'
]

ALL_STARS = [
    'Jrue Holiday', 'Derrick White', 'Pascal Siakam', 'Bam Adebayo',
    'Jimmy Butler', 'Jalen Brunson', 'De\'Aaron Fox', 'Domantas Sabonis',
    'Karl-Anthony Towns', 'Chet Holmgren', 'Alperen Sengun'
]


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class BalldontlieCollector:
    """
    Collecteur de données via l'API Balldontlie.
    
    Récupère blessures, cotes et stats pour les matchs du jour.
    """
    
    BASE_URL = "https://api.balldontlie.io/v1"
    
    def __init__(self, api_key: str = None):
        """
        Initialise le collecteur.
        
        Args:
            api_key: Clé API Balldontlie (ou variable d'env BALLDONTLIE_API_KEY)
        """
        self.api_key = api_key or os.environ.get('BALLDONTLIE_API_KEY', '')
        
        if not self.api_key:
            print("⚠️ Pas de clé API Balldontlie. Certaines fonctions seront limitées.")
            print("   Crée un compte gratuit sur https://www.balldontlie.io/")
        
        self.headers = {
            "Authorization": self.api_key
        }
        
        self.cache = {}
        self.cache_time = {}
        self.rate_limit_remaining = 5  # Tier gratuit: 5 req/min
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Fait une requête à l'API avec gestion du rate limiting.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params,
                timeout=30
            )
            
            # Vérifier le rate limit
            if 'X-RateLimit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
            
            if response.status_code == 429:
                print("⚠️ Rate limit atteint, attente 60s...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur API: {e}")
            return {}
    
    def _get_cached(self, key: str, ttl_minutes: int = 30) -> Optional[any]:
        """Récupère une valeur du cache si elle n'est pas expirée."""
        if key in self.cache and key in self.cache_time:
            age = datetime.now() - self.cache_time[key]
            if age < timedelta(minutes=ttl_minutes):
                return self.cache[key]
        return None
    
    def _set_cache(self, key: str, value: any):
        """Met en cache une valeur."""
        self.cache[key] = value
        self.cache_time[key] = datetime.now()
    
    # =========================================================================
    # MATCHS DU JOUR
    # =========================================================================
    
    def get_today_games(self) -> pd.DataFrame:
        """
        Récupère les matchs NBA du jour.
        
        Returns:
            DataFrame avec: id, date, home_team, away_team, status, scores
        """
        cache_key = 'today_games'
        cached = self._get_cached(cache_key, ttl_minutes=10)
        if cached is not None:
            return cached
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        data = self._make_request('games', {'dates[]': today})
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        games = []
        for game in data['data']:
            games.append({
                'game_id': game['id'],
                'date': game['date'],
                'status': game['status'],
                'home_team_id': game['home_team']['id'],
                'home_team': game['home_team']['full_name'],
                'home_team_abbr': game['home_team']['abbreviation'],
                'away_team_id': game['visitor_team']['id'],
                'away_team': game['visitor_team']['full_name'],
                'away_team_abbr': game['visitor_team']['abbreviation'],
                'home_score': game.get('home_team_score', 0),
                'away_score': game.get('visitor_team_score', 0),
            })
        
        df = pd.DataFrame(games)
        self._set_cache(cache_key, df)
        return df
    
    # =========================================================================
    # BLESSURES
    # =========================================================================
    
    def get_injuries(self, team_id: int = None) -> pd.DataFrame:
        """
        Récupère les blessures actives.
        
        Args:
            team_id: Filtrer par équipe (optionnel)
        
        Returns:
            DataFrame avec: player_name, team, status, description
        """
        cache_key = f'injuries_{team_id or "all"}'
        cached = self._get_cached(cache_key, ttl_minutes=30)
        if cached is not None:
            return cached
        
        params = {}
        if team_id:
            params['team_ids[]'] = team_id
        
        data = self._make_request('injuries', params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        injuries = []
        for inj in data['data']:
            player = inj.get('player', {})
            team = inj.get('team', {})
            
            injuries.append({
                'player_id': player.get('id'),
                'player_name': f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                'team_id': team.get('id'),
                'team': team.get('full_name', ''),
                'team_abbr': team.get('abbreviation', ''),
                'status': inj.get('status', ''),
                'description': inj.get('description', ''),
                'start_date': inj.get('start_date', ''),
                'return_date': inj.get('return_date', ''),
            })
        
        df = pd.DataFrame(injuries)
        self._set_cache(cache_key, df)
        return df
    
    def get_today_injuries(self) -> pd.DataFrame:
        """Alias pour get_injuries()"""
        return self.get_injuries()
    
    def get_team_injuries(self, team_name: str, injuries_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Filtre les blessures pour une équipe.
        
        Args:
            team_name: Nom de l'équipe (ex: "Boston Celtics" ou "Celtics")
        """
        if injuries_df is None:
            injuries_df = self.get_injuries()
        
        if injuries_df.empty:
            return pd.DataFrame()
        
        # Matching flexible
        team_lower = team_name.lower()
        mask = (
            injuries_df['team'].str.lower().str.contains(team_lower, na=False) |
            injuries_df['team'].str.lower().str.contains(team_lower.split()[-1], na=False)
        )
        
        return injuries_df[mask]
    
    def get_player_tier(self, player_name: str) -> str:
        """Détermine le tier d'importance d'un joueur."""
        if player_name in SUPERSTARS:
            return 'superstar'
        elif player_name in ALL_STARS:
            return 'allstar'
        else:
            return 'starter'  # Par défaut
    
    def calculate_injury_impact(self, team_name: str, injuries_df: pd.DataFrame = None) -> Dict:
        """
        Calcule l'impact des blessures sur une équipe.
        
        Returns:
            Dict avec total_impact, out_players, key_absences
        """
        team_injuries = self.get_team_injuries(team_name, injuries_df)
        
        if team_injuries.empty:
            return {
                'total_impact': 0.0,
                'out_players': [],
                'questionable_players': [],
                'key_absences': [],
                'injury_score': 0
            }
        
        out_players = []
        questionable_players = []
        key_absences = []
        total_impact = 0.0
        
        out_statuses = ['Out', 'out', 'OUT']
        questionable_statuses = ['Questionable', 'Day-To-Day', 'Probable', 'GTD']
        
        for _, row in team_injuries.iterrows():
            player = row['player_name']
            status = row['status']
            desc = row.get('description', '')
            
            tier = self.get_player_tier(player)
            impact = PLAYER_IMPACT.get(tier, 0.01)
            
            if status in out_statuses:
                out_players.append({
                    'name': player,
                    'status': status,
                    'description': desc,
                    'tier': tier,
                    'impact': impact
                })
                total_impact += impact
                
                if tier in ['superstar', 'allstar']:
                    key_absences.append(player)
                    
            elif any(s.lower() in status.lower() for s in questionable_statuses):
                questionable_players.append({
                    'name': player,
                    'status': status,
                    'description': desc,
                    'tier': tier,
                    'impact': impact * 0.4  # 40% car incertain
                })
                total_impact += impact * 0.3
        
        # Score normalisé 0-100
        injury_score = min(total_impact * 1000, 100)
        
        return {
            'total_impact': total_impact,
            'out_players': out_players,
            'questionable_players': questionable_players,
            'key_absences': key_absences,
            'injury_score': injury_score
        }
    
    def get_matchup_injury_diff(
        self, 
        home_team: str, 
        away_team: str,
        injuries_df: pd.DataFrame = None
    ) -> Dict:
        """
        Calcule le différentiel de blessures entre deux équipes.
        
        Un différentiel positif = avantage pour l'équipe à domicile.
        """
        if injuries_df is None:
            injuries_df = self.get_injuries()
        
        home_impact = self.calculate_injury_impact(home_team, injuries_df)
        away_impact = self.calculate_injury_impact(away_team, injuries_df)
        
        # Plus l'adversaire est touché, mieux c'est
        differential = away_impact['total_impact'] - home_impact['total_impact']
        
        # Résumé textuel
        summary_parts = []
        if home_impact['key_absences']:
            summary_parts.append(f"🏠 Sans: {', '.join(home_impact['key_absences'])}")
        if away_impact['key_absences']:
            summary_parts.append(f"🏃 Sans: {', '.join(away_impact['key_absences'])}")
        
        summary = ' | '.join(summary_parts) if summary_parts else "✅ Pas d'absences majeures"
        
        return {
            'home_impact': home_impact,
            'away_impact': away_impact,
            'differential': differential,
            'summary': summary
        }
    
    # =========================================================================
    # COTES (ODDS)
    # =========================================================================
    
    def get_today_odds(self) -> pd.DataFrame:
        """
        Récupère les cotes pour les matchs du jour.
        
        Returns:
            DataFrame avec: game_id, home_team, away_team, spread, moneyline, total
        """
        cache_key = 'today_odds'
        cached = self._get_cached(cache_key, ttl_minutes=15)
        if cached is not None:
            return cached
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        data = self._make_request('odds', {'dates[]': today})
        
        if not data or 'data' not in data:
            # Fallback: essayer via les games
            return self._get_odds_from_games()
        
        odds_list = []
        for odd in data['data']:
            game = odd.get('game', {})
            
            # Parser les différentes lignes
            spread_home = odd.get('spread_home')
            spread_away = odd.get('spread_away')
            moneyline_home = odd.get('moneyline_home')
            moneyline_away = odd.get('moneyline_away')
            total = odd.get('over_under')
            
            # Calculer les probabilités implicites
            prob_home = self._moneyline_to_prob(moneyline_home) if moneyline_home else None
            prob_away = self._moneyline_to_prob(moneyline_away) if moneyline_away else None
            
            odds_list.append({
                'game_id': game.get('id'),
                'home_team': game.get('home_team', {}).get('full_name', ''),
                'away_team': game.get('visitor_team', {}).get('full_name', ''),
                'spread_home': spread_home,
                'spread_away': spread_away,
                'moneyline_home': moneyline_home,
                'moneyline_away': moneyline_away,
                'total': total,
                'implied_prob_home': prob_home,
                'implied_prob_away': prob_away,
            })
        
        df = pd.DataFrame(odds_list)
        self._set_cache(cache_key, df)
        return df
    
    def _get_odds_from_games(self) -> pd.DataFrame:
        """Fallback: construire les odds depuis les games si endpoint dédié indisponible."""
        games = self.get_today_games()
        if games.empty:
            return pd.DataFrame()
        
        # Sans endpoint odds, retourner un DataFrame vide avec les bonnes colonnes
        return pd.DataFrame(columns=[
            'game_id', 'home_team', 'away_team', 
            'spread_home', 'moneyline_home', 'total',
            'implied_prob_home', 'implied_prob_away'
        ])
    
    def _moneyline_to_prob(self, ml: int) -> float:
        """
        Convertit une cote moneyline américaine en probabilité implicite.
        
        Ex: -150 → 60%, +200 → 33.3%
        """
        if ml is None:
            return None
        
        try:
            ml = int(ml)
            if ml < 0:
                return abs(ml) / (abs(ml) + 100)
            else:
                return 100 / (ml + 100)
        except:
            return None
    
    def match_odds(self, home_team: str, away_team: str, odds_df: pd.DataFrame = None) -> Tuple[float, float]:
        """
        Trouve les cotes pour un matchup spécifique.
        
        Returns:
            Tuple (implied_prob_home, implied_prob_away) ou (None, None)
        """
        if odds_df is None:
            odds_df = self.get_today_odds()
        
        if odds_df.empty:
            return None, None
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        for _, row in odds_df.iterrows():
            h_match = (
                home_lower in row['home_team'].lower() or 
                home_team.split()[-1].lower() in row['home_team'].lower()
            )
            a_match = (
                away_lower in row['away_team'].lower() or
                away_team.split()[-1].lower() in row['away_team'].lower()
            )
            
            if h_match and a_match:
                return row.get('implied_prob_home'), row.get('implied_prob_away')
        
        return None, None


# =============================================================================
# FONCTION HELPER POUR L'APP
# =============================================================================

def get_injury_adjustment(home_team: str, away_team: str, api_key: str = None) -> Tuple[float, str]:
    """
    Fonction simple pour obtenir l'ajustement de blessures.
    
    Returns:
        Tuple (adjustment, summary)
    """
    try:
        collector = BalldontlieCollector(api_key=api_key)
        matchup = collector.get_matchup_injury_diff(home_team, away_team)
        return matchup['differential'], matchup['summary']
    except Exception as e:
        return 0.0, f"⚠️ Blessures non disponibles"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏀 TEST - Balldontlie API")
    print("=" * 60)
    
    api_key = os.environ.get('BALLDONTLIE_API_KEY', '')
    
    if not api_key:
        print("\n⚠️ Définis ta clé API:")
        print("   export BALLDONTLIE_API_KEY='ta_clé'")
        print("   ou crée un compte sur https://www.balldontlie.io/")
    
    collector = BalldontlieCollector(api_key=api_key)
    
    # Test matchs du jour
    print("\n📅 Matchs du jour:")
    games = collector.get_today_games()
    if not games.empty:
        print(games[['home_team', 'away_team', 'status']].to_string())
    else:
        print("   Aucun match trouvé")
    
    # Test blessures
    print("\n🏥 Blessures actives:")
    injuries = collector.get_injuries()
    if not injuries.empty:
        print(f"   {len(injuries)} joueurs blessés")
        print(injuries[['player_name', 'team', 'status']].head(10).to_string())
    else:
        print("   Aucune blessure trouvée")
    
    # Test impact
    if not injuries.empty and not games.empty:
        print("\n📊 Test impact blessures:")
        home = games.iloc[0]['home_team']
        away = games.iloc[0]['away_team']
        diff = collector.get_matchup_injury_diff(home, away, injuries)
        print(f"   {away} @ {home}")
        print(f"   Différentiel: {diff['differential']:+.1%}")
        print(f"   {diff['summary']}")
