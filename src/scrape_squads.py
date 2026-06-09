import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import urllib3

# Désactiver les avertissements de sécurité liés au SSL ignoré
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. Dictionnaire de traduction FR -> EN (Standard Transfermarkt)
COUNTRY_TRANSLATION = {
    "États-Unis": "United States", "Mexique": "Mexico", "Arabie saoudite": "Saudi Arabia",
    "Corée du Sud": "South Korea", "Afrique du Sud": "South Africa", "Algérie": "Algeria",
    "Cap-Vert": "Cape Verde", "Côte d'Ivoire": "Ivory Coast", "Égypte": "Egypt",
    "Maroc": "Morocco", "RD Congo": "DR Congo", "Sénégal": "Senegal", "Tunisie": "Tunisia",
    "Curaçao": "Curaçao", "Haïti": "Haiti", "Panama": "Panama", "Argentine": "Argentina",
    "Brésil": "Brazil", "Colombie": "Colombia", "Équateur": "Ecuador", "Uruguay": "Uruguay",
    "Nouvelle-Zélande": "New Zealand", "Allemagne": "Germany", "Angleterre": "England",
    "Autriche": "Austria", "Belgique": "Belgium", "Bosnie-Herzégovine": "Bosnia and Herzegovina",
    "Croatie": "Croatia", "Écosse": "Scotland", "Espagne": "Spain", "Norvège": "Norway",
    "Pays-Bas": "Netherlands", "Suède": "Sweden", "Suisse": "Switzerland", "Tchéquie": "Czechia",
    "Turquie": "Turkey", "Japon": "Japan", "Australie": "Australia", "Iran": "Iran",
    "Irak": "Iraq", "Ouzbékistan": "Uzbekistan", "Qatar": "Qatar", "Jordanie": "Jordan",
    "Canada": "Canada", "France": "France", "Portugal": "Portugal", "Paraguay": "Paraguay"
}

def scrape_wikipedia_squads():
    print("=" * 60)
    print("🕸️ ORACLE MPP : WEBSCRAPING DES EFFECTIFS OFFICIELS (SSL BYPASS)")
    print("=" * 60)
    
    url = "https://fr.wikipedia.org/wiki/Effectif_des_équipes_à_la_Coupe_du_monde_de_football_2026"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    print(f"📡 Connexion à Wikipédia en cours (sans vérification SSL)...")
    
    try:
        # L'AJOUT CRUCIAL EST ICI : verify=False
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Erreur critique de connexion : {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    squad_data = []
    current_country = None
    
    for element in soup.find_all(['h2', 'h3', 'table']):
        if element.name in ['h2', 'h3']:
            title_text = element.get_text(strip=True).replace('[modifier | modifier le code]', '')
            for fr_name, en_name in COUNTRY_TRANSLATION.items():
                if fr_name in title_text:
                    current_country = en_name
                    break
                    
        elif element.name == 'table' and 'sortable' in element.get('class', []) and current_country:
            df_table = pd.read_html(str(element))[0]
            
            name_col = None
            for col in df_table.columns:
                if 'Nom' in str(col) or 'Joueur' in str(col):
                    name_col = col
                    break
            
            if name_col:
                for idx, row in df_table.iterrows():
                    player_name = str(row[name_col]).replace('(cap.)', '').strip()
                    if player_name and player_name != 'nan':
                        squad_data.append({
                            'Team': current_country,
                            'Player_Name': player_name
                        })
            current_country = None

    final_df = pd.DataFrame(squad_data)
    
    if final_df.empty:
        print("❌ La page Wikipédia ne contient pas (encore) les tableaux au format standard, ou n'existe pas.")
        return
        
    teams_found = final_df['Team'].nunique()
    print(f"✅ Scraping terminé ! {len(final_df)} joueurs extraits pour {teams_found} nations.")
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/official_squads_2026.csv'
    final_df.to_csv(out_path, index=False)
    print(f"💾 Fichier sauvegardé : {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    scrape_wikipedia_squads()