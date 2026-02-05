import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
import json
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ta clé API - remplace par la tienne ou utilise une variable d'environnement
API_KEY = os.environ.get('ODDS_API_KEY', '5f6dbc5b14e4790b5b1968b88b27c581')

# URL de base de l'API
BASE_URL = "https://api.the-odds-api.com/v4"

# Sport NBA
SPORT = "basketball_nba"

# Bookmakers par ordre de fiabilité (Pinnacle = référence du marché)
PREFERRED_BOOKMAKERS = [
    'pinnacle',      # Le plus sharp (référence)
    'fanduel',       # Gros volume US
    'draftkings',    # Gros volume US
    'betmgm',        # Gros volume US
    'williamhill_us', # Historique solide
    'bovada',        # Populaire
]

# Fichier de cache pour éviter de gaspiller les requêtes
CACHE_FILE = 'data/odds_cache.json'


# =============================================================================
# FONCTIONS DE CONVERSION DES COTES
# =============================================================================

def american_to_probability(odds: float) -> float:
    """
    Convertit une cote américaine en probabilité implicite.
    
    Exemples:
        -150 → 60% (favori)
        +130 → 43.5% (underdog)
        -200 → 66.7%
        +200 → 33.3%
    """
    if odds < 0:
        # Favori: -150 signifie miser 150 pour gagner 100
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog: +130 signifie miser 100 pour gagner 130
        return 100 / (odds + 100)


def decimal_to_probability(odds: float) -> float:
    """
    Convertit une cote décimale en probabilité implicite.
    
    Exemples:
        1.50 → 66.7%
        2.00 → 50%
        3.00 → 33.3%
    """
    return 1 / odds


def remove_vig(prob_home: float, prob_away: float) -> Tuple[float, float]:
    """
    Supprime la marge du bookmaker (vig/juice) pour obtenir
    les "vraies" probabilités.
    
    Les bookmakers ajoutent une marge (~4-5% en NBA) pour garantir leur profit.
    Exemple: Home 55% + Away 50% = 105% (5% de marge)
    
    On normalise pour revenir à 100%.
    """
    total = prob_home + prob_away
    if total > 0:
        return prob_home / total, prob_away / total
    return 0.5, 0.5


# =============================================================================
# COLLECTE DES COTES
# =============================================================================

class OddsCollector:
    """
    Collecteur de cotes NBA.
    
    Usage:
        collector = OddsCollector()
        
        # Cotes du jour
        odds = collector.get_upcoming_odds()
        
        # Pour un match spécifique
        prob = collector.get_match_probability("Lakers", "Celtics")
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        self.cache = self._load_cache()
        self._requests_remaining = None
        self._requests_used = None
    
    def _load_cache(self) -> Dict:
        """Charge le cache des cotes."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️ Erreur lecture cache: {e}")
                return {}
        return {}
    
    def _save_cache(self, data: List[Dict]):
        """
        Sauvegarde le cache en convertissant les dates en strings.
        
        FIX: Les Timestamp pandas ne sont pas JSON serializable,
        donc on les convertit en strings avant de sauvegarder.
        """
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            
            # Convertir les dates en strings pour la sérialisation JSON
            serializable_data = []
            for record in data:
                clean_record = {}
                for key, value in record.items():
                    # Convertir les timestamps/dates en strings
                    if isinstance(value, (pd.Timestamp, datetime)):
                        clean_record[key] = value.isoformat()
                    elif hasattr(value, 'isoformat'):  # date objects
                        clean_record[key] = value.isoformat()
                    elif pd.isna(value):
                        clean_record[key] = None
                    else:
                        clean_record[key] = value
                serializable_data.append(clean_record)
            
            cache_key = f"upcoming_{datetime.now().strftime('%Y%m%d_%H')}"
            self.cache[cache_key] = serializable_data
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
                
        except Exception as e:
            print(f"   ⚠️ Erreur sauvegarde cache: {e}")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Effectue une requête à l'API."""
        
        if self.api_key == 'YOUR_API_KEY_HERE' or not self.api_key:
            print("⚠️  API_KEY non configurée!")
            print("   1. Va sur https://the-odds-api.com/")
            print("   2. Crée un compte gratuit")
            print("   3. Copie ta clé API")
            print("   4. Remplace dans src/odds_collector.py")
            return None
        
        url = f"{BASE_URL}/{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Tracker les requêtes restantes
            self._requests_remaining = response.headers.get('x-requests-remaining')
            self._requests_used = response.headers.get('x-requests-used')
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("❌ Clé API invalide")
                return None
            elif response.status_code == 429:
                print("❌ Limite de requêtes atteinte (500/mois)")
                return None
            else:
                print(f"❌ Erreur API: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Timeout de la requête API")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur requête: {e}")
            return None
    
    def get_api_usage(self) -> Tuple[Optional[str], Optional[str]]:
        """Retourne le nombre de requêtes utilisées/restantes."""
        return self._requests_used, self._requests_remaining
    
    def get_upcoming_odds(self, days_ahead: int = 1) -> pd.DataFrame:
        """
        Récupère les cotes des matchs à venir.
        
        Args:
            days_ahead: Nombre de jours à récupérer
        
        Returns:
            DataFrame avec les cotes de chaque match
        """
        print(f"📊 Récupération des cotes NBA...")
        
        # Vérifier le cache (valide 1 heure)
        cache_key = f"upcoming_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.cache:
            print("   (depuis le cache)")
            cached_data = self.cache[cache_key]
            df = pd.DataFrame(cached_data)
            if not df.empty and 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'])
                df['game_date'] = df['commence_time'].dt.date
            return df
        
        # Requête API
        data = self._make_request(
            f"sports/{SPORT}/odds",
            params={
                'regions': 'us',
                'markets': 'h2h',  # Moneyline (vainqueur du match)
                'oddsFormat': 'american',
            }
        )
        
        if not data:
            return pd.DataFrame()
        
        # Parser les résultats
        matches = []
        
        for event in data:
            match_info = {
                'game_id': event.get('id'),
                'commence_time': event.get('commence_time'),  # String ISO format
                'home_team': event.get('home_team'),
                'away_team': event.get('away_team'),
            }
            
            # Extraire les cotes de chaque bookmaker
            best_home_prob = 0
            best_away_prob = 0
            odds_count = 0
            
            for bookmaker in event.get('bookmakers', []):
                book_name = bookmaker.get('key')
                
                # Chercher les cotes h2h (vainqueur)
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        outcomes = {o['name']: o['price'] for o in market.get('outcomes', [])}
                        
                        home_odds = outcomes.get(match_info['home_team'])
                        away_odds = outcomes.get(match_info['away_team'])
                        
                        if home_odds and away_odds:
                            # Convertir en probabilités
                            home_prob = american_to_probability(home_odds)
                            away_prob = american_to_probability(away_odds)
                            
                            # Supprimer la marge
                            home_prob, away_prob = remove_vig(home_prob, away_prob)
                            
                            # Garder les meilleures cotes (Pinnacle si dispo)
                            if book_name in PREFERRED_BOOKMAKERS[:3]:
                                match_info[f'{book_name}_home_prob'] = home_prob
                                match_info[f'{book_name}_away_prob'] = away_prob
                                
                                if book_name == 'pinnacle' or best_home_prob == 0:
                                    best_home_prob = home_prob
                                    best_away_prob = away_prob
                            
                            odds_count += 1
            
            # Stocker les meilleures probabilités
            match_info['implied_prob_home'] = best_home_prob
            match_info['implied_prob_away'] = best_away_prob
            match_info['bookmakers_count'] = odds_count
            
            if odds_count > 0:
                matches.append(match_info)
        
        df = pd.DataFrame(matches)
        
        if not df.empty:
            # Convertir la date
            df['commence_time'] = pd.to_datetime(df['commence_time'])
            df['game_date'] = df['commence_time'].dt.date
            
            # Sauvegarder en cache (avec conversion des dates)
            self._save_cache(df.to_dict('records'))
            
            print(f"   ✅ {len(df)} matchs avec cotes récupérés")
            
            # Afficher usage API
            used, remaining = self.get_api_usage()
            if used and remaining:
                print(f"   📈 API: {used} requêtes utilisées, {remaining} restantes")
        else:
            print("   ⚠️ Aucun match avec cotes trouvé")
        
        return df
    
    def get_match_probability(
        self, 
        home_team: str, 
        away_team: str,
        game_date: datetime = None
    ) -> Optional[Tuple[float, float]]:
        """
        Récupère la probabilité implicite pour un match spécifique.
        
        Args:
            home_team: Nom de l'équipe à domicile
            away_team: Nom de l'équipe à l'extérieur
            game_date: Date du match (défaut: aujourd'hui)
        
        Returns:
            Tuple (prob_home, prob_away) ou None si non trouvé
        """
        df = self.get_upcoming_odds()
        
        if df.empty:
            return None
        
        # Chercher le match (correspondance partielle sur les noms)
        for _, row in df.iterrows():
            home_match = home_team.lower() in row['home_team'].lower() or \
                        row['home_team'].lower() in home_team.lower()
            away_match = away_team.lower() in row['away_team'].lower() or \
                        row['away_team'].lower() in away_team.lower()
            
            if home_match and away_match:
                return row['implied_prob_home'], row['implied_prob_away']
        
        return None


# =============================================================================
# INTÉGRATION AVEC LE DATASET
# =============================================================================

def add_odds_to_matchups(matchups_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les cotes des bookmakers au dataset de matchups.
    
    Args:
        matchups_df: Dataset de matchups existant
        odds_df: DataFrame des cotes récupérées
    
    Returns:
        Dataset enrichi avec les probabilités implicites
    """
    matchups_df = matchups_df.copy()
    
    # Ajouter les colonnes de cotes si elles n'existent pas
    if 'IMPLIED_PROB_HOME' not in matchups_df.columns:
        matchups_df['IMPLIED_PROB_HOME'] = np.nan
        matchups_df['IMPLIED_PROB_AWAY'] = np.nan
    
    return matchups_df


# =============================================================================
# MAPPING DES NOMS D'ÉQUIPES
# =============================================================================

TEAM_NAME_MAPPING = {
    'Los Angeles Lakers': 'Los Angeles Lakers',
    'Los Angeles Clippers': 'LA Clippers',
    'Brooklyn Nets': 'Brooklyn Nets',
    'New York Knicks': 'New York Knicks',
    'Golden State Warriors': 'Golden State Warriors',
    'Phoenix Suns': 'Phoenix Suns',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Boston Celtics': 'Boston Celtics',
    'Miami Heat': 'Miami Heat',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Denver Nuggets': 'Denver Nuggets',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Sacramento Kings': 'Sacramento Kings',
    'Dallas Mavericks': 'Dallas Mavericks',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'Atlanta Hawks': 'Atlanta Hawks',
    'Toronto Raptors': 'Toronto Raptors',
    'Chicago Bulls': 'Chicago Bulls',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Utah Jazz': 'Utah Jazz',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Indiana Pacers': 'Indiana Pacers',
    'Washington Wizards': 'Washington Wizards',
    'Orlando Magic': 'Orlando Magic',
    'Charlotte Hornets': 'Charlotte Hornets',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Detroit Pistons': 'Detroit Pistons',
    'Houston Rockets': 'Houston Rockets',
}


def normalize_team_name(name: str) -> str:
    """Normalise le nom d'équipe pour le matching."""
    return TEAM_NAME_MAPPING.get(name, name)


# =============================================================================
# TEST DU MODULE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎰 TEST DU COLLECTEUR DE COTES")
    print("=" * 60)
    
    collector = OddsCollector()
    
    # Récupérer les cotes
    odds = collector.get_upcoming_odds()
    
    if not odds.empty:
        print(f"\n📋 Matchs trouvés:")
        print("-" * 60)
        
        for _, row in odds.iterrows():
            home = row['home_team']
            away = row['away_team']
            prob_h = row['implied_prob_home']
            prob_a = row['implied_prob_away']
            
            # Déterminer le favori
            if prob_h > prob_a:
                fav = home
                fav_prob = prob_h
            else:
                fav = away
                fav_prob = prob_a
            
            print(f"\n{away} @ {home}")
            print(f"   Domicile: {prob_h:.1%} | Extérieur: {prob_a:.1%}")
            print(f"   → Favori: {fav} ({fav_prob:.1%})")
    else:
        print("\n⚠️  Aucune cote récupérée")
        print("   Vérifie ta clé API ou la période (hors-saison?)")