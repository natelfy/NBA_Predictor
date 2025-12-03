import pandas as pd
import numpy as np

def get_elo_season(df):
    """Calcule du Elo Rating match par match."""
    df = df.sort_values(['GAME_DATE', 'GAME_ID'])

    dicoElo = {team: 1500 for team in df['TEAM_ID'].unique()}

    df['ELO_PRE'] = 0.0

    k_factor = 20

    for index, row in df.iterrows():
        idTeam = row['TEAM_ID']

        currentElo = dicoElo[idTeam]
        df.at[index, 'ELO_PRE'] = currentElo

        margin = row['PLUS_MINUS'] if not pd.isna(row['PLUS_MINUS']) else 0
        res = 1 if row['WL'] == 'W' else 0

        change = k_factor * (res - 0.5) + (margin * 0.1)
        dicoElo[idTeam] += change
    
    return df

def process_data():
    print("Traitement des données...")

    try:
        df = pd.read_csv('data/raw_games.csv')
    except FileNotFoundError:
        print("Erreur : Lance get_data.py d'abord !")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')

    # Tirs convertis en points
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']

    # Possessions / Turnovers
    df['POSS_EST'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    df['TOV_PCT'] = df['TOV'] / df['POSS_EST']

    #FT convertis
    df['FT_RATE'] = df['FTM'] / df['FGA']

    #Rebond offensif approximatif
    df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB'])

    #Calcul Elo
    print(" Calcul de l'ELO... ")
    df = get_elo_season(df)

    #Calcul de la fatigue
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days.fillna(3).clip(0, 7)

    #Victoire
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

    #Moyennes
    cols_to_roll = ['EFG_PCT', 'TOV_PCT', 'FT_RATE', 'OREB_PCT', 'PTS']
    for col in cols_to_roll:
        df[f'AVG_{col}_10'] = df.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).rolling(10).mean())

    final_cols = ['TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'WIN', 'IS_HOME', 'REST_DAYS', 'ELO_PRE'] + [f'AVG_{c}_10' for c in cols_to_roll]
    df = df.dropna(subset=final_cols)

    df.to_csv('data/processed_games.csv', index=False)
    print("Données prêtes")

if __name__ == "__main__":
    process_data()