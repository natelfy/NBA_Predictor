import pandas as pd
import numpy as np
import os

def process_pipeline():
    print("=" * 60)
    print("🧠 ORACLE MPP : RECALIBRAGE ABSOLU DE L'ELO (CORRECTION SYMÉTRIQUE)")
    print("=" * 60)

    raw_path = 'data/raw_international_games.csv'
    live_path = 'data/wc_2026_results.csv'

    if not os.path.exists(raw_path):
        print(f"❌ Fichier historique introuvable : {raw_path}")
        return

    print("⏳ Chargement de la base de données historique...")
    df_hist = pd.read_csv(raw_path)
    df_hist['MATCH_DATE'] = pd.to_datetime(df_hist['MATCH_DATE'])

    # 1. INJECTION DU MOMENTUM
    if os.path.exists(live_path):
        print("🔥 Injection des résultats de la Coupe du Monde 2026...")
        df_live = pd.read_csv(live_path)
        df_live['date'] = pd.to_datetime(df_live['date'])

        home_df = df_live[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
        home_df.columns = ['MATCH_DATE', 'TEAM_ID', 'OPP_TEAM_ID', 'GOALS', 'OPP_GOALS']
        home_df['IS_HOST'] = 1
        home_df['TOURNAMENT_CATEGORY'] = 'World Cup'

        away_df = df_live[['date', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
        away_df.columns = ['MATCH_DATE', 'TEAM_ID', 'OPP_TEAM_ID', 'GOALS', 'OPP_GOALS']
        away_df['IS_HOST'] = 0
        away_df['TOURNAMENT_CATEGORY'] = 'World Cup'

        df_hist = pd.concat([df_hist, home_df, away_df], ignore_index=True)

    df_hist = df_hist.sort_values('MATCH_DATE').reset_index(drop=True)

    print("🧮 Traitement du véritable ELO Symétrique...")
    
    # Création d'une empreinte unique pour chaque match (Date + Équipes triées)
    # Cela empêche l'algorithme de calculer deux fois le même match
    df_hist['MATCH_HASH'] = df_hist.apply(
        lambda x: str(x['MATCH_DATE']) + "_" + "_".join(sorted([str(x['TEAM_ID']), str(x['OPP_TEAM_ID'])])), 
        axis=1
    )
    
    elo_dict = {}
    row_elos = np.zeros(len(df_hist))
    processed_hashes = set()

    def get_k_factor(tournament):
        if pd.isna(tournament): return 20
        t = str(tournament)
        if 'World Cup' in t: return 50
        elif 'Continental' in t or 'Copa' in t or 'Euro' in t: return 40
        elif 'Qualifier' in t: return 30
        else: return 20

    # LA BOUCLE SYMÉTRIQUE
    for idx, row in df_hist.iterrows():
        m_hash = row['MATCH_HASH']
        team = str(row['TEAM_ID'])
        opp = str(row['OPP_TEAM_ID'])
        
        if team not in elo_dict: elo_dict[team] = 1500
        if opp not in elo_dict: elo_dict[opp] = 1500

        if m_hash not in processed_hashes:
            elo_team_pre = elo_dict[team]
            elo_opp_pre = elo_dict[opp]
            
            goals = row['GOALS']
            opp_goals = row['OPP_GOALS']
            
            if goals > opp_goals:
                actual_team, actual_opp = 1, 0
            elif goals == opp_goals:
                actual_team, actual_opp = 0.5, 0.5
            else:
                actual_team, actual_opp = 0, 1
                
            expected_team = 1 / (1 + 10 ** ((elo_opp_pre - elo_team_pre) / 400))
            expected_opp = 1 - expected_team
            
            k = get_k_factor(row.get('TOURNAMENT_CATEGORY', 'Other'))
            
            # Mise à jour simultanée (Le secret de l'équilibre)
            elo_dict[team] = elo_team_pre + k * (actual_team - expected_team)
            elo_dict[opp] = elo_opp_pre + k * (actual_opp - expected_opp)
            
            processed_hashes.add(m_hash)
        
        # On assigne le véritable niveau de l'équipe à cette ligne
        row_elos[idx] = elo_dict[team]

    df_hist['ELO_POST'] = row_elos

    print("📈 Modélisation des Expected Goals récents...")
    df_hist = df_hist.sort_values(['TEAM_ID', 'MATCH_DATE'])
    
    df_hist['AVG_GOALS_10'] = df_hist.groupby('TEAM_ID')['GOALS'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1)).fillna(1.2)
    df_hist['AVG_OPP_GOALS_10'] = df_hist.groupby('TEAM_ID')['OPP_GOALS'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1)).fillna(1.2)

    df_hist = df_hist.sort_values(['MATCH_DATE', 'MATCH_ID'], na_position='first')

    out_path = 'data/processed_football_games.csv'
    df_hist.to_csv(out_path, index=False)

    print(f"✅ Recalibrage parfait terminé. La hiérarchie mondiale est restaurée.")
    print("=" * 60)

if __name__ == "__main__":
    process_pipeline()