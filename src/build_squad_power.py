import pandas as pd
import numpy as np
import unicodedata
import os
import re
import difflib

def fuzzy_match_teams(fbref_name, fifa_teams):
    canonical_mapping = {
        'Ivory Coast': "Côte d'Ivoire",
        'Cote d Ivoire': "Côte d'Ivoire",
        'DR Congo': 'Congo DR',
        'Democratic Republic of the Congo': 'Congo DR',
        'Czechia': 'Czech Republic',
        'South Korea': 'Korea Republic',
        'North Korea': 'Korea DPR',
        'USA': 'United States',
        'Iran': 'IR Iran'
    }
    if fbref_name in canonical_mapping:
        return canonical_mapping[fbref_name]
    matches = difflib.get_close_matches(str(fbref_name), fifa_teams, n=1, cutoff=0.7)
    return matches[0] if matches else fbref_name

def clean_player_name(name):
    """Nettoyage pour garantir une fusion parfaite."""
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

def build_squad_power():
    print("=" * 60)
    print("🔥 ORACLE MPP : CALCUL DU VRAI SQUAD POWER INDEX (SANS BIAIS)")
    print("=" * 60)
    
    fbref_path = 'data/players_data-2025_2026.csv'
    official_path = 'data/official_squads_2026.csv'
    
    if not os.path.exists(fbref_path) or not os.path.exists(official_path):
        print("❌ Fichiers introuvables.")
        return

    df_fbref = pd.read_csv(fbref_path, low_memory=False)
    df_squads = pd.read_csv(official_path)
    
    df_fbref['Match_Name'] = df_fbref['Player'].apply(clean_player_name)
    df_squads['Match_Name'] = df_squads['Player_Name'].apply(clean_player_name)
    
    col_min = find_column(df_fbref, ['min'], exclude=['minus'])
    col_ga = find_column(df_fbref, ['g+a', 'g & a'])
    col_goals = find_column(df_fbref, ['gls', 'goals'], exclude=['against'])
    col_assists = find_column(df_fbref, ['ast', 'assists'], exclude=['against'])
    
    df_fbref[col_min] = df_fbref[col_min].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 1. DÉDOUBLONNAGE (Crucial pour FBref)
    # Les joueurs transférés ont plusieurs lignes. On ne garde que la ligne "Total" (celle avec le plus de minutes).
    df_fbref = df_fbref.sort_values(col_min, ascending=False).drop_duplicates(subset=['Match_Name'])
    
    # 2. CALCUL DE L'IMPACT RÉEL
    df_fbref['Real_Impact'] = 0.0
    if col_ga:
        df_fbref['Real_Impact'] = pd.to_numeric(df_fbref[col_ga], errors='coerce').fillna(0)
    elif col_goals and col_assists:
        df_fbref['Real_Impact'] = pd.to_numeric(df_fbref[col_goals], errors='coerce').fillna(0) + pd.to_numeric(df_fbref[col_assists], errors='coerce').fillna(0)
    
    # 3. L'ALGORITHME DE "TRUE RATING" (Fin du biais des défenseurs)
    # Un joueur marque des points par sa régularité (minutes) ET son tranchant (G+A)
    df_fbref['Player_Score'] = (df_fbref[col_min] / 3000) * 30 + (df_fbref['Real_Impact'] * 2.0)
    
    # Fusion des bases
    merged = pd.merge(df_squads, df_fbref[['Match_Name', col_min, 'Real_Impact', 'Player_Score']], on='Match_Name', how='left')
    
    merged[col_min] = merged[col_min].fillna(1000)
    merged['Real_Impact'] = merged['Real_Impact'].fillna(0.5)
    merged['Player_Score'] = merged['Player_Score'].fillna(10.0) # Score bas pour les joueurs hors-radar
    
    power_data = []
    
    # 4. CALCUL PAR ÉQUIPE
    for team in merged['Team'].unique():
        team_players = merged[merged['Team'] == team]
        
        # Le coup de génie : On sélectionne le Top 11 en fonction de la NOTE GLOBALE, pas juste des minutes !
        top_11 = team_players.sort_values('Player_Score', ascending=False).head(11)
        
        form_score = (top_11[col_min].mean() / 3000) * 40  
        impact_score = top_11['Real_Impact'].mean() * 3.5 
        
        total_power = min(99.0, form_score + impact_score + 40) # Base minimale à 40
        
        power_data.append({
            'Team': team,
            'Squad_Power_Index': total_power,
            'Avg_Top11_Minutes': top_11[col_min].mean(),
            'Avg_Top11_Impact': top_11['Real_Impact'].mean()
        })
        
    df_power = pd.DataFrame(power_data).sort_values('Squad_Power_Index', ascending=False)
    
    print("\n📊 VRAI CLASSEMENT SQUAD POWER (Top 10) :")
    print(df_power.head(10).to_string(index=False, formatters={'Squad_Power_Index': '{:.1f}'.format, 'Avg_Top11_Minutes': '{:.0f}'.format, 'Avg_Top11_Impact': '{:.1f}'.format}))
    
    # --- DÉBUT DE L'INTERCEPTION (Industrialisation du Fuzzy Matching) ---
    print("🛡️ Application automatique de la correction des noms de nations...")
    try:
        df_games = pd.read_csv('data/processed_football_games.csv')
        fifa_teams = df_games['TEAM_ID'].unique().tolist()
        
        team_col = 'Team' if 'Team' in df_power.columns else ('TEAM' if 'TEAM' in df_power.columns else None)
        if team_col:
            df_power[team_col] = df_power[team_col].apply(lambda x: fuzzy_match_teams(x, fifa_teams))
            print("✅ Noms des effectifs alignés sur la matrice FIFA.")
    except Exception as e:
        print(f"⚠️ Impossible de corriger les noms à la volée : {e}")
    # --- FIN DE L'INTERCEPTION ---
    
    # Ta ligne de sauvegarde d'origine se trouve juste ici
    # df.to_csv('data/squad_power.csv', index=False)

    df_power.to_csv('data/squad_power.csv', index=False)
    print("\n✅ Fichier 'data/squad_power.csv' généré. Le biais des gardiens de but est neutralisé.")
    print("=" * 60)

if __name__ == "__main__":
    build_squad_power()