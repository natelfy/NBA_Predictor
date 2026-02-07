"""
NBA Oracle - Tracking des Prédictions (CORRIGÉ v2)
===================================================

FIXES:
- Meilleur matching des noms d'équipes
- Debug logs pour diagnostic
- Gestion des cas où l'API ne retourne pas de données
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Optional, Dict, List, Tuple

# NBA API
try:
    from nba_api.stats.endpoints import scoreboardv2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

TRACKING_FILE = 'data/predictions_history.csv'

# Mapping des noms d'équipes (ton format → format NBA API)
TEAM_NAME_VARIANTS = {
    # Format: 'mot_clé': 'nom_complet_nba_api'
    'lakers': 'Lakers',
    'celtics': 'Celtics',
    'warriors': 'Warriors',
    'nets': 'Nets',
    'knicks': 'Knicks',
    'bulls': 'Bulls',
    'heat': 'Heat',
    'bucks': 'Bucks',
    'sixers': '76ers',
    '76ers': '76ers',
    'suns': 'Suns',
    'mavericks': 'Mavericks',
    'mavs': 'Mavericks',
    'clippers': 'Clippers',
    'nuggets': 'Nuggets',
    'jazz': 'Jazz',
    'blazers': 'Trail Blazers',
    'trail blazers': 'Trail Blazers',
    'timberwolves': 'Timberwolves',
    'wolves': 'Timberwolves',
    'pelicans': 'Pelicans',
    'grizzlies': 'Grizzlies',
    'spurs': 'Spurs',
    'rockets': 'Rockets',
    'thunder': 'Thunder',
    'kings': 'Kings',
    'raptors': 'Raptors',
    'pacers': 'Pacers',
    'hawks': 'Hawks',
    'hornets': 'Hornets',
    'wizards': 'Wizards',
    'magic': 'Magic',
    'pistons': 'Pistons',
    'cavaliers': 'Cavaliers',
    'cavs': 'Cavaliers',
}


def normalize_team_name(name: str) -> str:
    """Extrait le nom d'équipe normalisé."""
    if not name:
        return ""
    
    name_lower = name.lower().strip()
    
    # Chercher dans les variantes connues
    for key, value in TEAM_NAME_VARIANTS.items():
        if key in name_lower:
            return value.lower()
    
    # Sinon, prendre le dernier mot (ex: "Los Angeles Lakers" → "lakers")
    return name_lower.split()[-1]


def teams_match(team1: str, team2: str) -> bool:
    """Vérifie si deux noms d'équipes correspondent."""
    norm1 = normalize_team_name(team1)
    norm2 = normalize_team_name(team2)
    
    if not norm1 or not norm2:
        return False
    
    # Match exact ou contenu
    return norm1 == norm2 or norm1 in norm2 or norm2 in norm1


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class PredictionTracker:
    """Gestionnaire de suivi des prédictions."""
    
    def __init__(self):
        self.history = self._load_history()
    
    def _load_history(self) -> pd.DataFrame:
        """Charge l'historique des prédictions."""
        if os.path.exists(TRACKING_FILE):
            try:
                df = pd.read_csv(TRACKING_FILE)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                return df
            except Exception as e:
                print(f"Erreur chargement historique: {e}")
                return self._create_empty_history()
        else:
            return self._create_empty_history()
    
    def _create_empty_history(self) -> pd.DataFrame:
        """Crée un DataFrame vide avec les bonnes colonnes."""
        return pd.DataFrame(columns=[
            'date', 'game_id', 'home_team', 'away_team',
            'pred_home_prob', 'pred_winner', 'pred_confidence',
            'market_home_prob', 'value_bet', 'value_edge',
            'actual_winner', 'home_score', 'away_score',
            'correct', 'resolved'
        ])
    
    def _save_history(self):
        """Sauvegarde l'historique."""
        os.makedirs('data', exist_ok=True)
        self.history.to_csv(TRACKING_FILE, index=False)
    
    def save_predictions(self, predictions: List[Dict]):
        """Sauvegarde les prédictions du jour."""
        today = datetime.now().date()
        today_ts = pd.Timestamp(today)
        
        for pred in predictions:
            pred_home_prob = pred['pred_home_prob']
            pred_winner = pred['home_team'] if pred_home_prob > 0.5 else pred['away_team']
            pred_confidence = max(pred_home_prob, 1 - pred_home_prob)
            
            market_prob = pred.get('market_home_prob')
            value_bet = False
            value_edge = 0.0
            
            if market_prob is not None and not pd.isna(market_prob):
                edge = pred_home_prob - market_prob
                if abs(edge) >= 0.05:
                    value_bet = True
                    value_edge = edge
            
            new_row = {
                'date': today_ts,
                'game_id': str(pred.get('game_id', '')),
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'pred_home_prob': float(pred_home_prob),
                'pred_winner': pred_winner,
                'pred_confidence': float(pred_confidence),
                'market_home_prob': float(market_prob) if market_prob is not None and not pd.isna(market_prob) else None,
                'value_bet': value_bet,
                'value_edge': float(value_edge),
                'actual_winner': None,
                'home_score': None,
                'away_score': None,
                'correct': None,
                'resolved': False
            }
            
            # Éviter les doublons
            if len(self.history) > 0 and 'date' in self.history.columns:
                self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
                
                # Match par date + équipes (plus fiable que game_id)
                mask = (self.history['date'] == today_ts) & \
                       (self.history['home_team'] == pred['home_team']) & \
                       (self.history['away_team'] == pred['away_team'])
                
                if mask.any():
                    for col in ['pred_home_prob', 'pred_winner', 'pred_confidence', 
                               'market_home_prob', 'value_bet', 'value_edge']:
                        self.history.loc[mask, col] = new_row[col]
                    continue
            
            self.history = pd.concat([
                self.history, 
                pd.DataFrame([new_row])
            ], ignore_index=True)
        
        self._save_history()
        print(f"✅ {len(predictions)} prédictions sauvegardées pour le {today}")
    
    def update_results(self, date: datetime = None):
        """Met à jour les résultats des matchs."""
        if not NBA_API_AVAILABLE:
            print("❌ NBA API non disponible")
            return
        
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        target_date = date.date() if hasattr(date, 'date') else date
        target_ts = pd.Timestamp(target_date)
        
        print(f"🔄 Mise à jour des résultats pour le {target_date}...")
        
        if len(self.history) == 0:
            print("   Aucune prédiction dans l'historique")
            return
        
        self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
        
        # Matchs non résolus de cette date
        mask = (self.history['date'].dt.date == target_date) & \
               (self.history['resolved'] == False)
        
        pending = self.history[mask]
        
        if len(pending) == 0:
            print("   Aucun match en attente pour cette date")
            return
        
        print(f"   {len(pending)} matchs en attente")
        
        # Récupérer les résultats via NBA API
        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=target_date.strftime('%Y-%m-%d'),
                timeout=30
            )
            games = board.game_header.get_data_frame()
            line_scores = board.line_score.get_data_frame()
            
            print(f"   {len(games)} matchs trouvés dans l'API")
            
        except Exception as e:
            print(f"   ⚠️ Erreur API: {e}")
            return
        
        if games.empty:
            print("   Aucun match retourné par l'API")
            return
        
        updated = 0
        
        for idx, row in pending.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            print(f"   Recherche: {away_team} @ {home_team}")
            
            # Chercher dans les résultats
            for _, game in games.iterrows():
                game_id = game['GAME_ID']
                game_scores = line_scores[line_scores['GAME_ID'] == game_id]
                
                if len(game_scores) < 2:
                    continue
                
                # Récupérer les infos des deux équipes
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['VISITOR_TEAM_ID']
                
                home_row = game_scores[game_scores['TEAM_ID'] == home_team_id]
                away_row = game_scores[game_scores['TEAM_ID'] == away_team_id]
                
                if len(home_row) == 0 or len(away_row) == 0:
                    continue
                
                # Construire les noms d'équipe depuis l'API
                api_home_name = ""
                api_away_name = ""
                
                if 'TEAM_CITY' in home_row.columns and 'TEAM_NAME' in home_row.columns:
                    api_home_name = f"{home_row['TEAM_CITY'].values[0]} {home_row['TEAM_NAME'].values[0]}"
                    api_away_name = f"{away_row['TEAM_CITY'].values[0]} {away_row['TEAM_NAME'].values[0]}"
                elif 'TEAM_NAME' in home_row.columns:
                    api_home_name = home_row['TEAM_NAME'].values[0]
                    api_away_name = away_row['TEAM_NAME'].values[0]
                
                # Vérifier si c'est le bon match
                home_match = teams_match(home_team, api_home_name)
                away_match = teams_match(away_team, api_away_name)
                
                if home_match and away_match:
                    home_score = home_row['PTS'].values[0]
                    away_score = away_row['PTS'].values[0]
                    
                    if pd.isna(home_score) or pd.isna(away_score):
                        print(f"      ⏳ Match pas encore terminé")
                        continue
                    
                    home_score = int(home_score)
                    away_score = int(away_score)
                    
                    actual_winner = home_team if home_score > away_score else away_team
                    correct = (actual_winner == row['pred_winner'])
                    
                    self.history.loc[idx, 'actual_winner'] = actual_winner
                    self.history.loc[idx, 'home_score'] = home_score
                    self.history.loc[idx, 'away_score'] = away_score
                    self.history.loc[idx, 'correct'] = correct
                    self.history.loc[idx, 'resolved'] = True
                    
                    status = "✅" if correct else "❌"
                    print(f"      {status} {away_score}-{home_score} → {actual_winner.split()[-1]}")
                    
                    updated += 1
                    break
            else:
                print(f"      ⚠️ Match non trouvé dans l'API")
        
        self._save_history()
        print(f"\n📊 {updated}/{len(pending)} résultats mis à jour")
    
    def get_stats(self, days: int = 30) -> Dict:
        """Calcule les statistiques de performance."""
        
        if len(self.history) == 0:
            return {
                'total_predictions': 0,
                'message': 'Pas encore de données'
            }
        
        self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
        valid_dates = self.history['date'].notna()
        
        if not valid_dates.any():
            return {
                'total_predictions': 0,
                'message': 'Pas de dates valides'
            }
        
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        
        resolved_mask = (self.history['resolved'] == True)
        date_mask = (self.history['date'] >= cutoff)
        
        resolved = self.history[valid_dates & resolved_mask & date_mask].copy()
        
        if len(resolved) == 0:
            pending = self.history[valid_dates & ~resolved_mask]
            return {
                'total_predictions': 0,
                'pending': len(pending),
                'message': f'{len(pending)} prédictions en attente de résultat'
            }
        
        total = len(resolved)
        resolved['correct'] = resolved['correct'].astype(bool)
        correct = resolved['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        # Par niveau de confiance
        by_confidence = {}
        confidence_bins = [
            (0.50, 0.55, '50-55%'),
            (0.55, 0.60, '55-60%'),
            (0.60, 0.65, '60-65%'),
            (0.65, 0.70, '65-70%'),
            (0.70, 1.00, '70%+'),
        ]
        
        for low, high, label in confidence_bins:
            conf_mask = (resolved['pred_confidence'] >= low) & (resolved['pred_confidence'] < high)
            subset = resolved[conf_mask]
            if len(subset) > 0:
                by_confidence[label] = {
                    'count': int(len(subset)),
                    'correct': int(subset['correct'].sum()),
                    'accuracy': float(subset['correct'].mean())
                }
        
        # Value bets
        value_bets = resolved[resolved['value_bet'] == True]
        value_stats = None
        if len(value_bets) > 0:
            value_stats = {
                'count': int(len(value_bets)),
                'correct': int(value_bets['correct'].sum()),
                'accuracy': float(value_bets['correct'].mean()),
                'avg_edge': float(value_bets['value_edge'].abs().mean())
            }
        
        return {
            'period_days': days,
            'total_predictions': int(total),
            'correct': int(correct),
            'accuracy': float(accuracy),
            'by_confidence': by_confidence,
            'value_bets': value_stats,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_pending_predictions(self) -> pd.DataFrame:
        """Retourne les prédictions en attente."""
        if len(self.history) == 0:
            return pd.DataFrame()
        return self.history[self.history['resolved'] == False]
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Retourne les prédictions récentes."""
        if len(self.history) == 0:
            return pd.DataFrame()
        
        self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        
        valid = self.history['date'].notna()
        recent = self.history[valid & (self.history['date'] >= cutoff)]
        
        return recent.sort_values('date', ascending=False)
    
    def print_stats(self, days: int = 30):
        """Affiche les statistiques."""
        stats = self.get_stats(days)
        
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE - PRÉDICTIONS RÉELLES")
        print("=" * 50)
        
        if stats['total_predictions'] == 0:
            print(f"\n⚠️ {stats.get('message', 'Pas de données')}")
            if 'pending' in stats:
                print(f"   {stats['pending']} prédictions en attente")
            return
        
        print(f"\n📅 Période: {stats['period_days']} jours")
        print(f"🎯 Total: {stats['total_predictions']} prédictions")
        print(f"✅ Correctes: {stats['correct']}")
        print(f"📈 Accuracy: {stats['accuracy']:.1%}")
        
        if stats['by_confidence']:
            print(f"\n{'─' * 40}")
            print("PAR CONFIANCE")
            for label, data in stats['by_confidence'].items():
                print(f"   {label}: {data['accuracy']:.0%} ({data['count']} matchs)")
        
        if stats['value_bets']:
            print(f"\n{'─' * 40}")
            print("VALUE BETS")
            vb = stats['value_bets']
            print(f"   Accuracy: {vb['accuracy']:.0%} ({vb['count']} matchs)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Oracle - Tracking')
    parser.add_argument('--update', action='store_true', help='Mettre à jour les résultats')
    parser.add_argument('--stats', action='store_true', help='Afficher les stats')
    parser.add_argument('--days', type=int, default=30, help='Nombre de jours')
    parser.add_argument('--debug', action='store_true', help='Mode debug')
    
    args = parser.parse_args()
    
    tracker = PredictionTracker()
    
    if args.debug:
        print("📋 Prédictions en attente:")
        pending = tracker.get_pending_predictions()
        if len(pending) > 0:
            print(pending[['date', 'home_team', 'away_team', 'pred_winner']].to_string())
        else:
            print("   Aucune")
    elif args.update:
        tracker.update_results()
    elif args.stats:
        tracker.print_stats(days=args.days)
    else:
        tracker.print_stats()