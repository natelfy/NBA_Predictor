import pandas as pd
import numpy as np
import unicodedata
import os
import re

def clean_player_name(name):
    if pd.isna(name): return ""
    n = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    n = re.sub(r'[^a-z0-9\s]', '', n.lower())
    return ' '.join(n.split())

def find_column(df, keywords, exclude=[]):
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords) and not any(excl in col_lower for excl in exclude):
            return col
    return None

def build_fatigue_index():
    print("=" * 60)
    print("🔋 ORACLE MPP : CALCUL DE L'INDICE DE FATIGUE (SANS BIAIS)")
    print("=" * 60)

    fbref_path = 'data/players_data-2025_2026.csv'
    official_path = 'data/official_squads_2026.csv'

    if not os.path.exists(fbref_path) or not os.path.exists(official_path):
        print("❌ Fichiers introuvables. Lance generate_proxy_squads.py d'abord.")
        return

    df_fbref = pd.read_csv(fbref_path, low_memory=False)
    df_squads = pd.read_csv(official_path)

    df_fbref['Match_Name'] = df_fbref['Player'].apply(clean_player_name)
    df_squads['Match_Name'] = df_squads['Player_Name'].apply(clean_player_name)

    col_min = find_column(df_fbref, ['min'], exclude=['minus'])
    col_ga = find_column(df_fbref, ['g+a', 'g & a'])
    col_goals = find_column(df_fbref, ['gls', 'goals'], exclude=['against'])
    col_assists = find_column(df_fbref, ['ast', 'assists'], exclude=['against'])

    # Nettoyage des virgules (ex: "3,450")
    df_fbref[col_min] = df_fbref[col_min].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)

    # 1. Dédoublonnage des transferts (On garde le total)
    df_fbref = df_fbref.sort_values(col_min, ascending=False).drop_duplicates(subset=['Match_Name'])

    # 2. Calcul d'impact pour isoler les vrais joueurs cadres
    df_fbref['Real_Impact'] = 0.0
    if col_ga:
        df_fbref['Real_Impact'] = pd.to_numeric(df_fbref[col_ga], errors='coerce').fillna(0)
    elif col_goals and col_assists:
        df_fbref['Real_Impact'] = pd.to_numeric(df_fbref[col_goals], errors='coerce').fillna(0) + pd.to_numeric(df_fbref[col_assists], errors='coerce').fillna(0)

    # 3. True Rating (Le mix Temps de jeu / Impact)
    df_fbref['Player_Score'] = (df_fbref[col_min] / 3000) * 30 + (df_fbref['Real_Impact'] * 2.0)

    # Fusion
    merged = pd.merge(df_squads, df_fbref[['Match_Name', col_min, 'Player_Score']], on='Match_Name', how='left')

    merged[col_min] = merged[col_min].fillna(1000)
    merged['Player_Score'] = merged['Player_Score'].fillna(10.0)

    fatigue_data = []

    for team in merged['Team'].unique():
        team_players = merged[merged['Team'] == team]
        
        # SÉLECTION SANS BIAIS : On prend le Top 11 en fonction du True Rating (Stars offensives incluses)
        top_11 = team_players.sort_values('Player_Score', ascending=False).head(11)
        
        avg_minutes = top_11[col_min].mean()

        # Le Modulateur de Burnout
        if avg_minutes > 3600:
            status = "🔴 BURNOUT DETECTED"
            modifier = 0.90  # Malus
        elif avg_minutes > 3200:
            status = "🟠 FATIGUE ÉLEVÉE"
            modifier = 0.96
        elif avg_minutes < 1500:
            status = "🟡 MANQUE DE RYTHME"
            modifier = 0.95
        else:
            status = "🟢 FRAÎCHEUR OPTIMALE"
            modifier = 1.05  # Boost

        fatigue_data.append({
            'Team': team,
            'Top11_Avg_Minutes': round(avg_minutes, 0),
            'Physical_Status': status,
            'Power_Modifier': modifier
        })

    df_fatigue = pd.DataFrame(fatigue_data).sort_values('Top11_Avg_Minutes', ascending=False)

    print("\n📊 ÉTAT DE FATIGUE RÉEL DES FAVORIS (Saison 25/26) :")
    print(df_fatigue.head(15).to_string(index=False))

    df_fatigue.to_csv('data/fatigue_index.csv', index=False)
    print("\n✅ Fichier 'data/fatigue_index.csv' régénéré sans le biais des gardiens de but.")
    print("=" * 60)

if __name__ == "__main__":
    build_fatigue_index()