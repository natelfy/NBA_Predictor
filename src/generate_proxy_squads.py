import pandas as pd
import os

def generate_proxy_squads():
    print("=" * 60)
    print("🧬 ORACLE MPP : GÉNÉRATION DES EFFECTIFS PAR REVERSE ENGINEERING")
    print("=" * 60)
    
    fbref_path = 'data/players_data-2025_2026.csv'
    if not os.path.exists(fbref_path):
        print(f"❌ Fichier FBref introuvable : {fbref_path}")
        print("Assure-toi d'avoir placé le dataset 'players_data-2025_2026.csv' dans le dossier data/")
        return

    # Dictionnaire pour traduire les codes pays de FBref vers nos noms stricts
    NATION_CODES = {
        "FRA": "France", "ENG": "England", "ESP": "Spain", "GER": "Germany",
        "POR": "Portugal", "NED": "Netherlands", "ARG": "Argentina", "BRA": "Brazil",
        "BEL": "Belgium", "URU": "Uruguay", "COL": "Colombia", "SEN": "Senegal",
        "MAR": "Morocco", "USA": "United States", "CAN": "Canada", "JPN": "Japan",
        "KOR": "South Korea", "CRO": "Croatia", "SUI": "Switzerland", "DEN": "Denmark",
        "POL": "Poland", "MEX": "Mexico", "GHA": "Ghana", "CIV": "Ivory Coast", 
        "AUS": "Australia", "KSA": "Saudi Arabia", "ECU": "Ecuador", "TUN": "Tunisia", 
        "DZA": "Algeria", "AUT": "Austria", "NOR": "Norway", "TUR": "Turkey", 
        "SWE": "Sweden", "CZE": "Czechia", "SCO": "Scotland", "BIH": "Bosnia and Herzegovina", 
        "RSA": "South Africa", "PAN": "Panama", "IRN": "Iran", "EGY": "Egypt"
    }

    print("⏳ Lecture de la base de données FBref...")
    df = pd.read_csv(fbref_path)
    
    # Nettoyage des minutes (gérer les "3,450")
    df['Min'] = df['Min'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Extraction du code pays (FBref écrit souvent "fr FRA", on veut juste "FRA")
    df['Nation_Code'] = df['Nation'].astype(str).str.split(' ').str[-1].str.upper()
    
    # Traduction vers nos noms de base
    df['Team'] = df['Nation_Code'].map(NATION_CODES)
    
    # On filtre les joueurs qui ont une nation reconnue
    df_valid = df.dropna(subset=['Team']).copy()
    
    proxy_squads = []
    
    # Pour chaque équipe, on prend les 23 joueurs ayant le plus de temps de jeu (les cadres)
    for team in df_valid['Team'].unique():
        team_players = df_valid[df_valid['Team'] == team].sort_values('Min', ascending=False).head(23)
        for _, row in team_players.iterrows():
            proxy_squads.append({
                'Team': team,
                'Player_Name': row['Player'],
                'Pos': row.get('Pos', 'N/A')
            })
            
    df_squads = pd.DataFrame(proxy_squads)
    
    output_path = 'data/official_squads_2026.csv'
    df_squads.to_csv(output_path, index=False)
    
    print(f"✅ Effectifs générés avec succès !")
    print(f"🌍 Nombre de nations trouvées dans le Top 5 européen : {df_squads['Team'].nunique()}")
    print(f"👥 Nombre total de joueurs enregistrés : {len(df_squads)}")
    print(f"💾 Fichier sauvegardé sous : {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    generate_proxy_squads()