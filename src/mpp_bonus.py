import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

QUALIFIED_TEAMS = [
    'Canada', 'United States', 'Mexico', 'Saudi Arabia', 'Australia', 'Iraq', 'Japan', 'Jordan', 
    'Uzbekistan', 'Qatar', 'South Korea', 'Iran', 'South Africa', 'Algeria', 'Cape Verde', 
    'Ivory Coast', 'Côte d\'Ivoire', 'Egypt', 'Ghana', 'Morocco', 'DR Congo', 'Senegal', 'Tunisia',
    'Curaçao', 'Haiti', 'Panama', 'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
    'New Zealand', 'Germany', 'England', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia', 
    'Scotland', 'France', 'Spain', 'Norway', 'Netherlands', 'Portugal', 'Sweden', 'Switzerland', 
    'Czechia', 'Czech Republic', 'Turkey'
]

def get_latest_stats():
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    recent_date = df['MATCH_DATE'].max() - pd.Timedelta(days=730)
    df = df[df['MATCH_DATE'] >= recent_date]
    latest = df.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()
    return latest[latest['TEAM_ID'].isin(QUALIFIED_TEAMS)]

def generate_advanced_bonus_predictions():
    print("=" * 60)
    print("🏆 ORACLE MPP : PRÉDICTIONS BONUS (AVEC CORRECTION D'EFFECTIF)")
    print("=" * 60)
    
    try:
        clf = joblib.load('models/football_classifier.pkl')
        with open('models/football_features.txt', 'r') as f:
            features = f.read().split(',')
    except:
        print("❌ Modèles introuvables. Lance train_football_models.py d'abord.")
        return

    stats = get_latest_stats()
    top_5 = stats.nlargest(5, 'ELO_POST')
    base_features = [f.replace('_DIFF', '') for f in features if f != 'T1_IS_HOST']
    boss_stats = top_5[base_features].mean()
    
    power_rankings = []
    
    # 1. Calcul de base (Probabilité ELO)
    for _, team in stats.iterrows():
        pred_data = {}
        for feat in features:
            if feat == 'T1_IS_HOST':
                pred_data[feat] = [0] 
            else:
                base_feat = feat.replace('_DIFF', '')
                try:
                    pred_data[feat] = [team[base_feat] - boss_stats[base_feat]]
                except:
                    pred_data[feat] = [0]
                    
        X_pred = pd.DataFrame(pred_data)[features]
        win_prob = clf.predict_proba(X_pred)[0][2]
        
        power_rankings.append({
            'Team': team['TEAM_ID'],
            'ELO': team['ELO_POST'],
            'Form_10': team.get('FORM_10', 0),
            'Defense': team.get('AVG_OPP_GOALS_10', 0),
            'Base_Prob': win_prob
        })
        
    results = pd.DataFrame(power_rankings)
    
    # 2. INTÉGRATION DE LA QUALITÉ D'EFFECTIF (La Magie)
    squad_file = 'data/squad_metrics.csv'
    if os.path.exists(squad_file):
        print("📥 Dataset Squad Metrics détecté ! Application de la correction...")
        squad_data = pd.read_csv(squad_file)
        results = results.merge(squad_data, on='Team', how='left')
        
        # On remplit les équipes sans valeur (les petits pays) avec des notes très basses
        min_val = results['Squad_Value_M'].min() if pd.notna(results['Squad_Value_M'].min()) else 10.0
        med_age = 27.5
        results['Squad_Value_M'] = results['Squad_Value_M'].fillna(min_val * 0.1)
        results['Average_Age'] = results['Average_Age'].fillna(med_age)
        
        # Normalisation de la Valeur (0 à 1)
        max_val = results['Squad_Value_M'].max()
        results['Value_Score'] = results['Squad_Value_M'] / max_val
        
        # Pénalité de vieillissement : si âge moyen > 28, on pénalise lourdement.
        results['Age_Penalty'] = np.where(results['Average_Age'] > 28.0, (results['Average_Age'] - 28.0) * 0.15, 0)
        
        results['True_Power_Index'] = (results['Base_Prob'] * 0.8) + (results['Value_Score'] * 0.2) - results['Age_Penalty']
    else:
        print("⚠️ ERREUR : Fichier 'data/squad_metrics.csv' non trouvé. Reviens en arrière.")
        return
        
    results = results.sort_values('True_Power_Index', ascending=False)
    
    # --- AFFICHAGE ---
    print("\n🥇 LE VAINQUEUR DU TOURNOI (True Power Index)")
    winner = results.iloc[0]
    print(f"   Recommandation : {winner['Team'].upper()}")
    print(f"   Justification : Effet croisé optimal. Valeur: {winner['Squad_Value_M']}M€ | Âge: {winner['Average_Age']} ans | Base Prob: {winner['Base_Prob']:.1%}")

    print("\n🚀 L'ÉQUIPE SURPRISE (High Form, Medium ELO)")
    outsiders = results[results['ELO'] < results['ELO'].quantile(0.70)]
    surprise = outsiders.sort_values('Form_10', ascending=False).iloc[0]
    print(f"   Recommandation : {surprise['Team'].upper()}")
    print(f"   Justification : Hors radar (ELO {surprise['ELO']:.0f}), mais dynamique de {surprise['Form_10']*15:.1f} pts/15 sur 10 matchs.")

    print("\n🧱 LA PIRE ÉQUIPE (Low ELO, Pire Défense)")
    bottom_tier = results[results['ELO'] < results['ELO'].quantile(0.15)]
    worst = bottom_tier.sort_values('Defense', ascending=False).iloc[0]
    print(f"   Recommandation : {worst['Team'].upper()}")
    print(f"   Justification : ELO le plus faible ({worst['ELO']:.0f}), encaisse {worst['Defense']:.1f} buts/match.")

    print("\n👟 LE MEILLEUR BUTEUR (L'Équation du 'Focal Point')")
    top_3_teams = results.head(3)['Team'].tolist()
    print(f"   Le Top 3 pour atteindre le dernier carré (7 matchs joués) : {', '.join([t.upper() for t in top_3_teams])}")
    print("   Règle mathématique : Ne prends PAS forcément le buteur du Vainqueur. Prends l'attaquant 'Focal Point' de l'équipe du Top 3 qui monopolise le plus grand % des buts de sa sélection.")
    print(f"   Recommandation : Identifie qui de {top_3_teams[0].upper()}, {top_3_teams[1].upper()} ou {top_3_teams[2].upper()} a le système le plus dépendant d'un seul joueur (ex: Mbappé, Kane) et tire les penaltys. C'est ta cible.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_advanced_bonus_predictions()