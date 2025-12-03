from nba_api.stats.endpoints import leaguegamelog
import pandas as pd 
import time

def get_data(seasons=['2022-23', '2023-24', '2024-25', '2025-26']):
    all_games = []

    print("Loading...")
    for season in seasons :
        print(f"Loading season {season}...")
        try:
            log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
            df = log.get_data_frames()[0]
            df['SEASON_ID'] = season
            all_games.append(df)
            print(f"Saison {season} : {len(df)} lignes.")
        except Exception as e:
            print(f"Erreur saison {season} : {e}")
        time.sleep(1)

    if all_games:
        final_df = pd.concat(all_games, ignore_index=True)
        final_df.to_csv('data/raw_games.csv', index=False)
        print("Données sauvegardées dans data/raw_games.csv")
    
if __name__ == "__main__":
    get_data()