"""
NBA Oracle - Tracking des Prédictions
======================================

Ce module permet de :
1. Sauvegarder les prédictions AVANT les matchs
2. Récupérer les résultats APRÈS les matchs
3. Calculer les métriques de performance réelle
4. Générer des rapports

Usage:
    # Sauvegarder les prédictions du jour (matin)
    python src/tracker.py --save
    
    # Mettre à jour les résultats (soir/lendemain)
    python src/tracker.py --update
    
    # Afficher les statistiques
    python src/tracker.py --stats
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Optional, Dict, List, Tuple

# NBA API
try:
    from nba_api.stats.endpoints import scoreboardv2, leaguegamelog
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

TRACKING_FILE = 'data/predictions_history.csv'
DAILY_PREDICTIONS_FILE = 'data/daily_predictions.json'
STATS_FILE = 'data/tracking_stats.json'


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class PredictionTracker:
    """
    Gestionnaire de suivi des prédictions.
    
    Workflow quotidien:
    1. Matin: save_predictions() - Sauvegarde les prédictions du jour
    2. Soir/Lendemain: update_results() - Récupère les vrais résultats
    3. À tout moment: get_stats() - Calcule les performances
    """
    
    def __init__(self):
        self.history = self._load_history()
    
    def _load_history(self) -> pd.DataFrame:
        """Charge l'historique des prédictions."""
        if os.path.exists(TRACKING_FILE):
            df = pd.read_csv(TRACKING_FILE)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
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
                {
                    'game_id': str,
                    'home_team': str,
                    'away_team': str,
                    'pred_home_prob': float,
                    'market_home_prob': float (optionnel),
                }
        """
        today = datetime.now().date()
        
        for pred in predictions:
            # Calculer le gagnant prédit
            pred_home_prob = pred['pred_home_prob']
            pred_winner = pred['home_team'] if pred_home_prob > 0.5 else pred['away_team']
            pred_confidence = max(pred_home_prob, 1 - pred_home_prob)
            
            # Value bet?
            market_prob = pred.get('market_home_prob')
            value_bet = False
            value_edge = 0
            
            if market_prob:
                edge = pred_home_prob - market_prob
                if abs(edge) >= 0.05:
                    value_bet = True
                    value_edge = edge
            
            # Ajouter à l'historique
            new_row = {
                'date': today,
                'game_id': pred['game_id'],
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'pred_home_prob': pred_home_prob,
                'pred_winner': pred_winner,
                'pred_confidence': pred_confidence,
                'market_home_prob': market_prob,
                'value_bet': value_bet,
                'value_edge': value_edge,
                'actual_winner': None,
                'home_score': None,
                'away_score': None,
                'correct': None,
                'resolved': False
            }
            
            # Éviter les doublons
            mask = (self.history['date'] == pd.Timestamp(today)) & \
                   (self.history['game_id'] == pred['game_id'])
            
            if mask.any():
                # Mettre à jour
                for col, val in new_row.items():
                    if col not in ['actual_winner', 'home_score', 'away_score', 'correct', 'resolved']:
                        self.history.loc[mask, col] = val
            else:
                # Ajouter
                self.history = pd.concat([
                    self.history, 
                    pd.DataFrame([new_row])
                ], ignore_index=True)
        
        self._save_history()
        print(f"✅ {len(predictions)} prédictions sauvegardées pour le {today}")
    
    def update_results(self, date: datetime = None):
        """
        Met à jour les résultats des matchs.
        
        Args:
            date: Date à mettre à jour (défaut: hier)
        """
        if not NBA_API_AVAILABLE:
            print("❌ NBA API non disponible")
            return
        
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        target_date = date.date() if hasattr(date, 'date') else date
        
        print(f"🔄 Mise à jour des résultats pour le {target_date}...")
        
        # Récupérer les matchs non résolus de cette date
        mask = (self.history['date'] == pd.Timestamp(target_date)) & \
               (self.history['resolved'] == False)
        
        pending = self.history[mask]
        
        if len(pending) == 0:
            print("   Aucun match en attente pour cette date")
            return
        
        # Récupérer les résultats via NBA API
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
            game_id = row['game_id']
            
            # Chercher le match dans les résultats
            # On match sur les noms d'équipes car game_id peut différer
            home_team = row['home_team']
            away_team = row['away_team']
            
            for _, game in games.iterrows():
                # Trouver les scores dans line_score
                game_scores = line_scores[line_scores['GAME_ID'] == game['GAME_ID']]
                
                if len(game_scores) < 2:
                    continue
                
                # Identifier home et away
                home_row = game_scores[game_scores['TEAM_ID'] == game['HOME_TEAM_ID']]
                away_row = game_scores[game_scores['TEAM_ID'] == game['VISITOR_TEAM_ID']]
                
                if len(home_row) == 0 or len(away_row) == 0:
                    continue
                
                home_score = home_row['PTS'].values[0]
                away_score = away_row['PTS'].values[0]
                
                # Vérifier que le match est terminé
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
                
                # Matcher sur le nom d'équipe
                api_home = home_row['TEAM_ABBREVIATION'].values[0] if 'TEAM_ABBREVIATION' in home_row.columns else ''
                api_away = away_row['TEAM_ABBREVIATION'].values[0] if 'TEAM_ABBREVIATION' in away_row.columns else ''
                
                # Correspondance flexible
                home_match = (home_team.lower() in str(api_home).lower() or 
                             str(api_home).lower() in home_team.lower() or
                             home_team.split()[-1].lower() in str(api_home).lower())
                
                if home_match or game['HOME_TEAM_ID'] == game_id:
                    # Match trouvé!
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
        """
        Calcule les statistiques de performance.
        
        Args:
            days: Nombre de jours à analyser
        
        Returns:
            Dict avec les métriques
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        resolved = self.history[
            (self.history['resolved'] == True) &
            (self.history['date'] >= cutoff)
        ]
        
        if len(resolved) == 0:
            return {
                'total_predictions': 0,
                'message': 'Pas encore de données'
            }
        
        # Métriques globales
        total = len(resolved)
        correct = resolved['correct'].sum()
        accuracy = correct / total
        
        # Par niveau de confiance
        confidence_bins = [
            (0.50, 0.55, '50-55%'),
            (0.55, 0.60, '55-60%'),
            (0.60, 0.65, '60-65%'),
            (0.65, 0.70, '65-70%'),
            (0.70, 1.00, '70%+'),
        ]
        
        by_confidence = {}
        for low, high, label in confidence_bins:
            mask = (resolved['pred_confidence'] >= low) & (resolved['pred_confidence'] < high)
            subset = resolved[mask]
            if len(subset) > 0:
                by_confidence[label] = {
                    'count': len(subset),
                    'correct': subset['correct'].sum(),
                    'accuracy': subset['correct'].mean()
                }
        
        # Value bets
        value_bets = resolved[resolved['value_bet'] == True]
        value_stats = None
        if len(value_bets) > 0:
            value_stats = {
                'count': len(value_bets),
                'correct': value_bets['correct'].sum(),
                'accuracy': value_bets['correct'].mean(),
                'avg_edge': value_bets['value_edge'].abs().mean()
            }
        
        # Évolution par semaine
        resolved_copy = resolved.copy()
        resolved_copy['week'] = resolved_copy['date'].dt.isocalendar().week
        weekly = resolved_copy.groupby('week').agg({
            'correct': ['sum', 'count']
        })
        weekly.columns = ['correct', 'total']
        weekly['accuracy'] = weekly['correct'] / weekly['total']
        
        return {
            'period_days': days,
            'total_predictions': total,
            'correct': int(correct),
            'accuracy': accuracy,
            'by_confidence': by_confidence,
            'value_bets': value_stats,
            'weekly': weekly.to_dict('index'),
            'last_updated': datetime.now().isoformat()
        }
    
    def print_stats(self, days: int = 30):
        """Affiche les statistiques de façon lisible."""
        
        stats = self.get_stats(days)
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE DU MODÈLE - PRÉDICTIONS RÉELLES")
        print("=" * 60)
        
        if stats['total_predictions'] == 0:
            print("\n⚠️ Pas encore de prédictions trackées.")
            print("   Lance: python src/tracker.py --save")
            return
        
        print(f"\n📅 Période: {stats['period_days']} derniers jours")
        print(f"🎯 Total: {stats['total_predictions']} prédictions")
        print(f"✅ Correctes: {stats['correct']}")
        print(f"📈 Accuracy réelle: {stats['accuracy']:.1%}")
        
        # Par confiance
        if stats['by_confidence']:
            print(f"\n{'─' * 40}")
            print("📊 PAR NIVEAU DE CONFIANCE")
            print(f"{'─' * 40}")
            print(f"{'Confiance':<12} {'Matchs':>8} {'Correct':>8} {'Accuracy':>10}")
            print("-" * 40)
            
            for label, data in stats['by_confidence'].items():
                print(f"{label:<12} {data['count']:>8} {data['correct']:>8} {data['accuracy']:>9.1%}")
        
        # Value bets
        if stats['value_bets']:
            print(f"\n{'─' * 40}")
            print("💎 VALUE BETS")
            print(f"{'─' * 40}")
            vb = stats['value_bets']
            print(f"   Nombre: {vb['count']}")
            print(f"   Accuracy: {vb['accuracy']:.1%}")
            print(f"   Edge moyen: {vb['avg_edge']:.1%}")
        
        print("\n" + "=" * 60)
    
    def get_pending_predictions(self) -> pd.DataFrame:
        """Retourne les prédictions en attente de résultat."""
        return self.history[self.history['resolved'] == False]
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Retourne les prédictions récentes."""
        cutoff = datetime.now() - timedelta(days=days)
        return self.history[self.history['date'] >= cutoff].sort_values('date', ascending=False)


# =============================================================================
# INTÉGRATION AVEC L'APP
# =============================================================================

def save_today_predictions(model, team_stats, games, features_list, odds_data=None):
    """
    Fonction helper pour sauvegarder les prédictions depuis app.py
    
    Args:
        model: Le modèle de prédiction
        team_stats: DataFrame des stats d'équipe
        games: DataFrame des matchs du jour
        features_list: Liste des features
        odds_data: DataFrame des cotes (optionnel)
    """
    from datetime import datetime
    
    tracker = PredictionTracker()
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
        
        # Prédiction (utilise la même logique que app.py)
        # Simplifié ici - à adapter selon ta fonction predict_match
        pred = {
            'game_id': str(game.get('GAME_ID', '')),
            'home_team': home_name,
            'away_team': away_name,
            'pred_home_prob': 0.5,  # À remplacer par ta vraie prédiction
            'market_home_prob': None
        }
        
        predictions.append(pred)
    
    if predictions:
        tracker.save_predictions(predictions)
    
    return tracker


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Oracle - Tracking des prédictions')
    parser.add_argument('--save', action='store_true', help='Sauvegarder les prédictions du jour')
    parser.add_argument('--update', action='store_true', help='Mettre à jour les résultats')
    parser.add_argument('--stats', action='store_true', help='Afficher les statistiques')
    parser.add_argument('--days', type=int, default=30, help='Nombre de jours pour les stats')
    parser.add_argument('--date', type=str, help='Date spécifique (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    tracker = PredictionTracker()
    
    if args.save:
        print("⚠️ Pour sauvegarder les prédictions, utilise l'interface Streamlit")
        print("   ou intègre save_today_predictions() dans ton code.")
    
    elif args.update:
        date = None
        if args.date:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        tracker.update_results(date)
    
    elif args.stats:
        tracker.print_stats(days=args.days)
    
    else:
        # Par défaut: afficher les stats
        tracker.print_stats(days=args.days)
        
        # Montrer les prédictions en attente
        pending = tracker.get_pending_predictions()
        if len(pending) > 0:
            print(f"\n⏳ {len(pending)} prédictions en attente de résultat")


if __name__ == "__main__":
    main()
