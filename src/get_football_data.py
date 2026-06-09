import pandas as pd
import numpy as np
import os

def process_kaggle_results(file_path):
    """
    Transforme le dataset Kaggle brut en dataset orienté 'Machine Learning'
    Format cible : 1 ligne = 1 performance d'équipe.
    """
    print(f"📂 Chargement du dataset: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Fichier introuvable. Place results.csv dans {file_path}")

    df = pd.read_csv(file_path)
    
    # 1. Nettoyage initial
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score']) # Retirer les matchs annulés
    
    # 2. Création d'un MATCH_ID unique
    df = df.sort_values('date').reset_index(drop=True)
    df['MATCH_ID'] = df.index + 1
    
    print(f"📊 {len(df)} matchs historiques trouvés (depuis {df['date'].min().year}).")

    # 3. Transformation : 1 ligne par match -> 2 lignes par match (Team 1 & Team 2)
    # L'objectif est d'avoir le point de vue de chaque équipe pour calculer les moyennes glissantes.
    
    # --- Vue DOMICILE (Home) ---
    home_df = df[['MATCH_ID', 'date', 'tournament', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral']].copy()
    home_df.columns = ['MATCH_ID', 'MATCH_DATE', 'TOURNAMENT', 'TEAM_ID', 'OPP_TEAM_ID', 'GOALS', 'OPP_GOALS', 'neutral']
    # Si le match n'est pas sur terrain neutre, l'équipe à domicile est l'hôte
    home_df['IS_HOST'] = np.where(home_df['neutral'] == False, 1, 0)
    
    # --- Vue EXTÉRIEUR (Away) ---
    away_df = df[['MATCH_ID', 'date', 'tournament', 'away_team', 'home_team', 'away_score', 'home_score', 'neutral']].copy()
    away_df.columns = ['MATCH_ID', 'MATCH_DATE', 'TOURNAMENT', 'TEAM_ID', 'OPP_TEAM_ID', 'GOALS', 'OPP_GOALS', 'neutral']
    # L'équipe à l'extérieur n'est jamais l'hôte
    away_df['IS_HOST'] = 0
    
    # 4. Fusion des deux vues
    ml_df = pd.concat([home_df, away_df], ignore_index=True)
    ml_df = ml_df.drop(columns=['neutral'])
    
    # 5. Tri chronologique rigoureux (Crucial pour le calcul ELO)
    ml_df = ml_df.sort_values(['MATCH_DATE', 'MATCH_ID']).reset_index(drop=True)
    
    # 6. Catégorisation des tournois (Pour le K-Factor de l'ELO)
    # On rassemble la multitude de tournois Kaggle en grandes catégories d'enjeu
    def categorize_tournament(t):
        t = str(t).lower()
        if 'world cup' in t and 'qualification' not in t:
            return 'World Cup'
        elif 'world cup qualification' in t:
            return 'Qualifier'
        elif any(x in t for x in ['euro ', 'copa america', 'african cup of nations', 'asian cup', 'gold cup']):
            return 'Continental Cup'
        elif 'qualification' in t:
            return 'Qualifier'
        elif 'nations league' in t:
            return 'Nations League'
        elif 'friendly' in t:
            return 'Friendly'
        else:
            return 'Other'

    ml_df['TOURNAMENT_CATEGORY'] = ml_df['TOURNAMENT'].apply(categorize_tournament)
    
    # 7. Sauvegarde
    output_path = 'data/raw_international_games.csv'
    os.makedirs('data', exist_ok=True)
    ml_df.to_csv(output_path, index=False)
    
    print(f"✅ Transformation terminée !")
    print(f"💾 Fichier sauvegardé : {output_path} ({len(ml_df)} lignes, 1 match = 2 lignes)")
    print("-" * 40)
    print("Aperçu des données :")
    print(ml_df[['MATCH_DATE', 'TEAM_ID', 'GOALS', 'OPP_GOALS', 'OPP_TEAM_ID', 'IS_HOST']].tail(4))

if __name__ == "__main__":
    process_kaggle_results('data/results.csv')