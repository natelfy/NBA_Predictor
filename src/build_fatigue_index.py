import pandas as pd
import numpy as np
import unicodedata
import os

def normalize_name(name):
    if pd.isna(name): return ""
    n = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    return n.lower().strip()

def build_fatigue_index():
    print("=" * 60)
    print("🔋 ORACLE MPP : CALCUL DE L'INDICE DE FATIGUE (BURNOUT WC 2026)")
    print("=" * 60)
    
    fbref_path = 'data/players_data-2025_2026.csv'
    official_path = 'data/official_squads_2026.csv'
    
    if not os.path.exists(fbref_path) or not os.path.exists(official_path):
        print("❌ Fichiers manquants. Place players_data-2025_2026.csv et official_squads_2026.csv dans data/")
        return

    # Chargement
    df_fbref = pd.read_csv(fbref_path)
    df_squads = pd.read_csv(official_path)
    
    # Nettoyage pour la jointure
    df_fbref['Norm_Name'] = df_fbref['Player'].apply(normalize_name)
    df_squads['Norm_Name'] = df_squads['Player_Name'].apply(normalize_name)
    
    # Gestion des colonnes de minutes (nettoyage des virgules si FBref sort "3,450")
    df_fbref['Min'] = df_fbref['Min'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Jointure des effectifs officiels avec leur temps de jeu en club
    merged = pd.merge(df_squads, df_fbref[['Norm_Name', 'Min']], on='Norm_Name', how='left')
    
    # On remplit les non-trouvés avec une moyenne par défaut (ex: joueurs hors top 5 européen)
    merged['Min'] = merged['Min'].fillna(2000) 
    
    # ---------------------------------------------------------
    # CALCUL PAR NATION (On prend les 11 joueurs les plus utilisés)
    # ---------------------------------------------------------
    fatigue_data = []
    
    for team in merged['Team'].unique():
        team_players = merged[merged['Team'] == team].sort_values('Min', ascending=False)
        top_11 = team_players.head(11)
        
        avg_minutes = top_11['Min'].mean()
        
        # Le Modulateur de Burnout
        # Zone idéale : 2200 - 3200 minutes. 
        # Au-delà de 3500 : Danger de blessure/baisse de forme.
        if avg_minutes > 3600:
            status = "🔴 BURNOUT DETECTED"
            modifier = 0.90  # Baisse de 10% de la force de l'équipe
        elif avg_minutes > 3200:
            status = "🟠 FATIGUE ÉLEVÉE"
            modifier = 0.96
        elif avg_minutes < 1500:
            status = "🟡 MANQUE DE RYTHME"
            modifier = 0.95
        else:
            status = "🟢 FRAÎCHEUR OPTIMALE"
            modifier = 1.05  # Boost de 5%
            
        fatigue_data.append({
            'Team': team,
            'Top11_Avg_Minutes': round(avg_minutes, 0),
            'Physical_Status': status,
            'Power_Modifier': modifier
        })
        
    df_fatigue = pd.DataFrame(fatigue_data).sort_values('Top11_Avg_Minutes', ascending=False)
    
    print("\n📊 ÉTAT DE FATIGUE DES FAVORIS (Saison 25/26) :")
    print(df_fatigue.to_string(index=False))
    
    # Sauvegarde pour le moteur de match
    df_fatigue.to_csv('data/fatigue_index.csv', index=False)
    print("\n✅ Fichier 'data/fatigue_index.csv' généré. Ton moteur de match (mpp_batch.py) va pouvoir ajuster les cotes en lisant ce modificateur de puissance.")
    print("=" * 60)

if __name__ == "__main__":
    build_fatigue_index()