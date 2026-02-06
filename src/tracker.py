"""
NBA Oracle - Tracking des Prédictions (CORRIGÉ)
================================================

Ce module permet de :
1. Sauvegarder les prédictions AVANT les matchs
2. Récupérer les résultats APRÈS les matchs
3. Calculer les métriques de performance réelle
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
                # Convertir la colonne date en datetime
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
        """
        Sauvegarde les prédictions du jour.
        
        Args:
            predictions: Liste de dicts avec les prédictions
        """
        today = datetime.now().date()
        today_ts = pd.Timestamp(today)
        
        for pred in predictions:
            # Calculer le gagnant prédit
            pred_home_prob = pred['pred_home_prob']
            pred_winner = pred['home_team'] if pred_home_prob > 0.5 else pred['away_team']
            pred_confidence = max(pred_home_prob, 1 - pred_home_prob)
            
            # Value bet?
            market_prob = pred.get('market_home_prob')
            value_bet = False
            value_edge = 0.0
            
            if market_prob is not None and not pd.isna(market_prob):
                edge = pred_home_prob - market_prob
                if abs(edge) >= 0.05:
                    value_bet = True
                    value_edge = edge
            
            # Nouveau row
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
                # S'assurer que les dates sont comparables
                self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
                
                mask = (self.history['date'] == today_ts) & \
                       (self.history['game_id'] == str(pred.get('game_id', '')))
                
                if mask.any():
                    # Mettre à jour les colonnes de prédiction uniquement
                    for col in ['pred_home_prob', 'pred_winner', 'pred_confidence', 
                               'market_home_prob', 'value_bet', 'value_edge']:
                        self.history.loc[mask, col] = new_row[col]
                    continue
            
            # Ajouter nouvelle ligne
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
        
        # Convertir les dates
        self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
        
        # Matchs non résolus de cette date
        mask = (self.history['date'].dt.date == target_date) & \
               (self.history['resolved'] == False)
        
        pending = self.history[mask]
        
        if len(pending) == 0:
            print("   Aucun match en attente pour cette date")
            return
        
        # Récupérer les résultats
        try:
            board = scoreboardv2.ScoreboardV2(
                game_date=target_date.strftime('%Y-%m-%d'),
                timeout=30
            )
            games = board.game_header.get_data_frame()
            line_scores = board.line_score.get_data_frame()
        except Exception as e:
            print(f"   ⚠️ Erreur API: {e}")
            return
        
        updated = 0
        
        for idx, row in pending.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Chercher dans les résultats
            for _, game in games.iterrows():
                game_scores = line_scores[line_scores['GAME_ID'] == game['GAME_ID']]
                
                if len(game_scores) < 2:
                    continue
                
                home_row = game_scores[game_scores['TEAM_ID'] == game['HOME_TEAM_ID']]
                away_row = game_scores[game_scores['TEAM_ID'] == game['VISITOR_TEAM_ID']]
                
                if len(home_row) == 0 or len(away_row) == 0:
                    continue
                
                home_score = home_row['PTS'].values[0]
                away_score = away_row['PTS'].values[0]
                
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
                
                # Matching sur nom d'équipe
                api_home_name = home_row['TEAM_CITY'].values[0] + ' ' + home_row['TEAM_NAME'].values[0] \
                    if 'TEAM_CITY' in home_row.columns else str(home_row['TEAM_ID'].values[0])
                
                home_match = (home_team.split()[-1].lower() in api_home_name.lower())
                
                if home_match:
                    actual_winner = home_team if home_score > away_score else away_team
                    correct = (actual_winner == row['pred_winner'])
                    
                    self.history.loc[idx, 'actual_winner'] = actual_winner
                    self.history.loc[idx, 'home_score'] = int(home_score)
                    self.history.loc[idx, 'away_score'] = int(away_score)
                    self.history.loc[idx, 'correct'] = correct
                    self.history.loc[idx, 'resolved'] = True
                    
                    status = "✅" if correct else "❌"
                    print(f"   {status} {away_team} @ {home_team}: {int(away_score)}-{int(home_score)}")
                    
                    updated += 1
                    break
        
        self._save_history()
        print(f"\n📊 {updated} résultats mis à jour")
    
    def get_stats(self, days: int = 30) -> Dict:
        """Calcule les statistiques de performance."""
        
        # Vérifier si l'historique est vide
        if len(self.history) == 0:
            return {
                'total_predictions': 0,
                'message': 'Pas encore de données'
            }
        
        # S'assurer que les dates sont bien formatées
        self.history['date'] = pd.to_datetime(self.history['date'], errors='coerce')
        
        # Filtrer les lignes avec des dates valides
        valid_dates = self.history['date'].notna()
        
        if not valid_dates.any():
            return {
                'total_predictions': 0,
                'message': 'Pas de dates valides'
            }
        
        # Cutoff date
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        
        # Filtrer: résolu ET date >= cutoff
        resolved_mask = (self.history['resolved'] == True)
        date_mask = (self.history['date'] >= cutoff)
        
        resolved = self.history[valid_dates & resolved_mask & date_mask].copy()
        
        if len(resolved) == 0:
            # Compter les prédictions en attente
            pending = self.history[valid_dates & ~resolved_mask]
            return {
                'total_predictions': 0,
                'pending': len(pending),
                'message': f'{len(pending)} prédictions en attente de résultat'
            }
        
        # Métriques globales
        total = len(resolved)
        
        # S'assurer que 'correct' est booléen
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
    
    args = parser.parse_args()
    
    tracker = PredictionTracker()
    
    if args.update:
        tracker.update_results()
    elif args.stats:
        tracker.print_stats(days=args.days)
    else:
        tracker.print_stats()