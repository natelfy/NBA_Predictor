"""
NBA Oracle - TTFL Module
=========================

Collecteur de statistiques joueurs pour la TTFL.

Formule TTFL:
Score = PTS + REB + AST + STL + BLK + FGM + 3PM + FTM
      - (FGA-FGM) - (3PA-3PM) - (FTA-FTM) - TOV
      + Bonus (Triple-double: +3, Double-double: +1.5)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from typing import Optional, Dict, List, Tuple

# NBA API
try:
    from nba_api.stats.endpoints import (
        leaguegamelog,
        playergamelog,
        scoreboardv2,
        commonplayerinfo,
        leaguedashplayerstats,
        playerindex
    )
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api non installé: pip install nba_api")


# =============================================================================
# CONFIGURATION
# =============================================================================

TTFL_PLAYERS_CACHE = 'data/ttfl_players_cache.csv'
TTFL_STATS_CACHE = 'data/ttfl_stats_cache.json'
DECK_FILE = 'data/ttfl_deck.json'


# =============================================================================
# CALCUL DU SCORE TTFL
# =============================================================================

def calculate_ttfl_score(row: pd.Series) -> float:
    """
    Calcule le score TTFL d'un joueur pour un match.
    
    Formule officielle:
    Score = PTS + REB + AST + STL + BLK 
          + FGM + FG3M + FTM 
          - (FGA - FGM) - (FG3A - FG3M) - (FTA - FTM) - TOV
          + Bonus
    """
    # Stats positives
    pts = row.get('PTS', 0) or 0
    reb = row.get('REB', 0) or 0
    ast = row.get('AST', 0) or 0
    stl = row.get('STL', 0) or 0
    blk = row.get('BLK', 0) or 0
    
    # Tirs réussis (bonus)
    fgm = row.get('FGM', 0) or 0
    fg3m = row.get('FG3M', 0) or 0
    ftm = row.get('FTM', 0) or 0
    
    # Tirs ratés (malus)
    fga = row.get('FGA', 0) or 0
    fg3a = row.get('FG3A', 0) or 0
    fta = row.get('FTA', 0) or 0
    
    fg_missed = fga - fgm
    fg3_missed = fg3a - fg3m
    ft_missed = fta - ftm
    
    # Turnovers (malus)
    tov = row.get('TOV', 0) or 0
    
    # Score de base
    score = (pts + reb + ast + stl + blk 
             + fgm + fg3m + ftm 
             - fg_missed - fg3_missed - ft_missed - tov)
    
    # Bonus Double-double / Triple-double
    stats_10plus = sum([
        1 if pts >= 10 else 0,
        1 if reb >= 10 else 0,
        1 if ast >= 10 else 0,
        1 if stl >= 10 else 0,
        1 if blk >= 10 else 0,
    ])
    
    if stats_10plus >= 3:
        score += 3  # Triple-double
    elif stats_10plus >= 2:
        score += 1.5  # Double-double
    
    return score


# =============================================================================
# COLLECTEUR DE STATS
# =============================================================================

class TTFLStatsCollector:
    """
    Collecte les statistiques des joueurs NBA pour la TTFL.
    """
    
    def __init__(self):
        self.cache = self._load_cache()
        self.players_df = None
    
    def _load_cache(self) -> Dict:
        """Charge le cache."""
        if os.path.exists(TTFL_STATS_CACHE):
            try:
                with open(TTFL_STATS_CACHE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache."""
        os.makedirs('data', exist_ok=True)
        with open(TTFL_STATS_CACHE, 'w') as f:
            json.dump(self.cache, f)
    
    def get_all_players(self, season: str = '2024-25') -> pd.DataFrame:
        """
        Récupère la liste de tous les joueurs actifs avec leurs stats moyennes.
        """
        print("📊 Récupération des joueurs NBA...")
        
        cache_key = f"players_{season}"
        cache_date = self.cache.get(f"{cache_key}_date", "")
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Cache valide 24h
        if cache_date == today and os.path.exists(TTFL_PLAYERS_CACHE):
            print("   (depuis le cache)")
            return pd.read_csv(TTFL_PLAYERS_CACHE)
        
        if not NBA_API_AVAILABLE:
            print("   ❌ NBA API non disponible")
            return pd.DataFrame()
        
        try:
            # Stats de la saison
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                timeout=60
            )
            df = stats.get_data_frames()[0]
            time.sleep(0.5)
            
            # Calculer le score TTFL moyen
            df['TTFL_AVG'] = df.apply(calculate_ttfl_score, axis=1)
            
            # Garder les colonnes utiles
            cols_keep = [
                'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                'FTM', 'FTA', 'FT_PCT', 'TTFL_AVG'
            ]
            df = df[[c for c in cols_keep if c in df.columns]]
            
            # Filtrer: min 10 matchs et 15 min/match
            df = df[(df['GP'] >= 10) & (df['MIN'] >= 15)]
            
            # Trier par TTFL moyen
            df = df.sort_values('TTFL_AVG', ascending=False).reset_index(drop=True)
            
            # Sauvegarder
            df.to_csv(TTFL_PLAYERS_CACHE, index=False)
            self.cache[f"{cache_key}_date"] = today
            self._save_cache()
            
            print(f"   ✅ {len(df)} joueurs récupérés")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return pd.DataFrame()
    
    def get_player_recent_games(self, player_id: int, n_games: int = 10) -> pd.DataFrame:
        """
        Récupère les derniers matchs d'un joueur avec scores TTFL.
        """
        if not NBA_API_AVAILABLE:
            return pd.DataFrame()
        
        try:
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25',
                timeout=30
            )
            df = log.get_data_frames()[0]
            
            if df.empty:
                return df
            
            # Calculer score TTFL pour chaque match
            df['TTFL_SCORE'] = df.apply(calculate_ttfl_score, axis=1)
            
            # Derniers N matchs
            df = df.head(n_games)
            
            return df
            
        except Exception as e:
            print(f"   ⚠️ Erreur récup joueur {player_id}: {e}")
            return pd.DataFrame()
    
    def get_today_players(self) -> pd.DataFrame:
        """
        Récupère les joueurs qui jouent aujourd'hui avec leurs stats.
        """
        print("🏀 Récupération des joueurs du jour...")
        
        if not NBA_API_AVAILABLE:
            return pd.DataFrame()
        
        try:
            # Matchs du jour
            board = scoreboardv2.ScoreboardV2(
                game_date=datetime.now().strftime('%Y-%m-%d'),
                timeout=30
            )
            games = board.game_header.get_data_frame()
            
            if games.empty:
                print("   ⚠️ Pas de match aujourd'hui")
                return pd.DataFrame()
            
            # Équipes qui jouent
            team_ids = set(games['HOME_TEAM_ID'].tolist() + games['VISITOR_TEAM_ID'].tolist())
            
            # Charger tous les joueurs
            all_players = self.get_all_players()
            
            if all_players.empty:
                return pd.DataFrame()
            
            # Filtrer ceux qui jouent
            today_players = all_players[all_players['TEAM_ID'].isin(team_ids)].copy()
            
            # Ajouter info du match
            def get_opponent(row):
                team_id = row['TEAM_ID']
                for _, game in games.iterrows():
                    if game['HOME_TEAM_ID'] == team_id:
                        return game['VISITOR_TEAM_ID'], 'HOME'
                    elif game['VISITOR_TEAM_ID'] == team_id:
                        return game['HOME_TEAM_ID'], 'AWAY'
                return None, None
            
            today_players[['OPPONENT_ID', 'HOME_AWAY']] = today_players.apply(
                lambda x: pd.Series(get_opponent(x)), axis=1
            )
            
            print(f"   ✅ {len(today_players)} joueurs ce soir")
            
            return today_players.sort_values('TTFL_AVG', ascending=False)
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return pd.DataFrame()
    
    def get_player_vs_team_history(self, player_id: int, team_id: int, n_games: int = 5) -> List[float]:
        """
        Récupère l'historique des scores TTFL d'un joueur contre une équipe.
        """
        if not NBA_API_AVAILABLE:
            return []
        
        try:
            # Récupérer tous les matchs du joueur
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25',
                timeout=30
            )
            df = log.get_data_frames()[0]
            
            if df.empty:
                return []
            
            # Filtrer par adversaire
            # Note: MATCHUP contient "vs. XXX" ou "@ XXX"
            team_abbr = teams.find_team_by_id(team_id)
            if team_abbr:
                team_abbr = team_abbr['abbreviation']
                df_vs = df[df['MATCHUP'].str.contains(team_abbr, na=False)]
                
                if not df_vs.empty:
                    df_vs['TTFL_SCORE'] = df_vs.apply(calculate_ttfl_score, axis=1)
                    return df_vs['TTFL_SCORE'].head(n_games).tolist()
            
            return []
            
        except:
            return []


# =============================================================================
# GESTIONNAIRE DE DECK
# =============================================================================

class TTFLDeckManager:
    """
    Gère le "deck" TTFL: joueurs utilisés et cooldown de 30 jours.
    """
    
    def __init__(self):
        self.deck = self._load_deck()
    
    def _load_deck(self) -> Dict:
        """Charge le deck."""
        if os.path.exists(DECK_FILE):
            try:
                with open(DECK_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {'picks': [], 'season': '2024-25'}
        return {'picks': [], 'season': '2024-25'}
    
    def _save_deck(self):
        """Sauvegarde le deck."""
        os.makedirs('data', exist_ok=True)
        with open(DECK_FILE, 'w') as f:
            json.dump(self.deck, f, indent=2)
    
    def add_pick(self, player_id: int, player_name: str, date: str = None):
        """Ajoute un joueur au deck (utilisé)."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self.deck['picks'].append({
            'player_id': player_id,
            'player_name': player_name,
            'date': date
        })
        self._save_deck()
    
    def remove_pick(self, player_id: int, date: str):
        """Supprime un pick (en cas d'erreur)."""
        self.deck['picks'] = [
            p for p in self.deck['picks'] 
            if not (p['player_id'] == player_id and p['date'] == date)
        ]
        self._save_deck()
    
    def is_available(self, player_id: int, cooldown_days: int = 30) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si un joueur est disponible.
        
        Returns:
            (disponible, date_dernier_pick ou None)
        """
        cutoff = datetime.now() - timedelta(days=cooldown_days)
        
        for pick in self.deck['picks']:
            if pick['player_id'] == player_id:
                pick_date = datetime.strptime(pick['date'], '%Y-%m-%d')
                if pick_date >= cutoff:
                    return False, pick['date']
        
        return True, None
    
    def get_available_date(self, player_id: int, cooldown_days: int = 30) -> Optional[str]:
        """Retourne la date à partir de laquelle le joueur sera dispo."""
        for pick in self.deck['picks']:
            if pick['player_id'] == player_id:
                pick_date = datetime.strptime(pick['date'], '%Y-%m-%d')
                available_date = pick_date + timedelta(days=cooldown_days)
                if available_date > datetime.now():
                    return available_date.strftime('%Y-%m-%d')
        return None
    
    def get_recent_picks(self, days: int = 30) -> List[Dict]:
        """Retourne les picks récents."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        
        for pick in self.deck['picks']:
            pick_date = datetime.strptime(pick['date'], '%Y-%m-%d')
            if pick_date >= cutoff:
                pick['days_ago'] = (datetime.now() - pick_date).days
                pick['available_in'] = max(0, 30 - pick['days_ago'])
                recent.append(pick)
        
        return sorted(recent, key=lambda x: x['date'], reverse=True)
    
    def get_stats(self) -> Dict:
        """Statistiques du deck."""
        total_picks = len(self.deck['picks'])
        recent = self.get_recent_picks(30)
        
        return {
            'total_picks': total_picks,
            'picks_30_days': len(recent),
            'players_locked': len(recent),
            'players_used_season': len(set(p['player_id'] for p in self.deck['picks']))
        }
    
    def reset_deck(self, confirm: bool = False):
        """Réinitialise le deck (nouvelle saison)."""
        if confirm:
            self.deck = {'picks': [], 'season': '2024-25'}
            self._save_deck()
            return True
        return False


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏀 TEST MODULE TTFL")
    print("=" * 60)
    
    collector = TTFLStatsCollector()
    
    # Test: récupérer tous les joueurs
    players = collector.get_all_players()
    
    if not players.empty:
        print(f"\n📊 TOP 10 TTFL (moyenne saison):")
        print("-" * 50)
        for i, row in players.head(10).iterrows():
            print(f"{i+1:2d}. {row['PLAYER_NAME']:25s} {row['TTFL_AVG']:5.1f} pts")
    
    # Test: joueurs du jour
    print("\n")
    today = collector.get_today_players()
    
    if not today.empty:
        print(f"\n🏀 TOP 10 CE SOIR:")
        print("-" * 50)
        for i, row in today.head(10).iterrows():
            loc = "🏠" if row['HOME_AWAY'] == 'HOME' else "✈️"
            print(f"{i+1:2d}. {row['PLAYER_NAME']:25s} {row['TTFL_AVG']:5.1f} pts {loc}")
