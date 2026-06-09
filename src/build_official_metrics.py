import pandas as pd
import numpy as np
import os
import unicodedata
from datetime import datetime

def normalize_name(name):
    """Nettoie les noms pour forcer la correspondance (supprime accents, majuscules, espaces)"""
    if pd.isna(name):
        return ""
    # Retire les accents et caractères spéciaux
    n = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    return n.lower().strip()

def build_perfect_squad_metrics():
    print("=" * 60)
    print("🧬 ORACLE MPP : FUSION DES EFFECTIFS OFFICIELS & TRANSFERMARKT")
    print("=" * 60)
    
    tm_path = 'data/players.csv'          # Le gros fichier Kaggle
    official_path = 'data/official_squads_2026.csv'  # Ton fichier parfait
    
    if not os.path.exists(tm_path) or not os.path.exists(official_path):
        print("❌ Fichiers manquants. Vérifie que players.csv et official_squads_2026.csv sont dans data/")
        return

    print("📂 Chargement des bases de données...")
    tm_df = pd.read_csv(tm_path)
    official_df = pd.read_csv(official_path)
    
    # 1. Préparation Transfermarkt
    tm_df = tm_df.dropna(subset=['market_value_in_eur', 'date_of_birth'])
    tm_df['Normalized_Name'] = tm_df['name'].apply(normalize_name)
    
    # Calcul de l'âge au coup d'envoi
    wc_start_date = datetime(2026, 6, 11)
    tm_df['date_of_birth'] = pd.to_datetime(tm_df['date_of_birth'], errors='coerce')
    tm_df['Age_in_2026'] = (wc_start_date - tm_df['date_of_birth']).dt.days / 365.25
    
    # 2. Préparation Liste Officielle
    official_df['Normalized_Name'] = official_df['Player_Name'].apply(normalize_name)
    
    print("🔍 Début de la traque des joueurs (Matching)...")
    
    # 3. La Jointure (MERGE)
    # On croise la liste officielle avec Transfermarkt via le nom normalisé
    merged = pd.merge(official_df, tm_df[['Normalized_Name', 'market_value_in_eur', 'Age_in_2026']], 
                      on='Normalized_Name', how='left')
    
    # Audit des joueurs non trouvés (Crucial pour la Data Science)
    missing_players = merged[merged['market_value_in_eur'].isna()]
    match_rate = 1 - (len(missing_players) / len(official_df))
    print(f"🎯 Taux de correspondance des noms : {match_rate:.1%}")
    if len(missing_players) > 0:
        print(f"⚠️ {len(missing_players)} joueurs non trouvés (valeur mise à zéro).")
    
    # 4. Aggrégation par équipe (La métrique finale)
    squad_metrics = merged.groupby('Team').agg(
        Squad_Value_M=('market_value_in_eur', lambda x: x.sum() / 1_000_000),
        Average_Age=('Age_in_2026', 'mean')
    ).reset_index()
    
    squad_metrics['Squad_Value_M'] = squad_metrics['Squad_Value_M'].round(1)
    squad_metrics['Average_Age'] = squad_metrics['Average_Age'].round(1)
    squad_metrics = squad_metrics.sort_values('Squad_Value_M', ascending=False)
    
    # Sauvegarde
    out_path = 'data/squad_metrics.csv'
    squad_metrics.to_csv(out_path, index=False)
    
    print(f"\n✅ Base de Qualité des Effectifs générée : {out_path}")
    print("\n🏆 TOP 5 DES EFFECTIFS OFFICIELS 2026 :")
    print(squad_metrics.head(5).to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    build_perfect_squad_metrics()