import pandas as pd
import numpy as np
from datetime import datetime

print("⏳ [1/3] Chargement des données brutes (Format Long détecté)...")
df = pd.read_csv('data/raw_international_games.csv')

# 1. Conversion des dates et tri chronologique absolu.
#    Tri aussi par IS_HOST décroissant (tri STABLE) : la ligne de l'équipe HÔTE devient T1
#    quand il y en a une -> orientation déterministe et T1_IS_HOST cohérent ; sinon ordre stable.
df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
df = df.sort_values(['MATCH_DATE', 'MATCH_ID', 'IS_HOST'],
                    ascending=[True, True, False], kind='mergesort').reset_index(drop=True)

# 2. Dédoublonnage : une seule ligne par MATCH_ID ("1 ligne = 1 match").
#    keep='first' -> T1 = équipe hôte si elle existe (sinon 1ère équipe, de façon stable et reproductible).
df_matches = df.drop_duplicates(subset=['MATCH_ID'], keep='first').copy()

# 3. Renommage tactique pour connecter au moteur de simulation
df_matches = df_matches.rename(columns={
    'MATCH_DATE': 'date',
    'TEAM_ID': 'home_team',
    'OPP_TEAM_ID': 'away_team',
    'GOALS': 'home_score',
    'OPP_GOALS': 'away_score',
    'IS_HOST': 'is_host_t1'
})

print(f"✅ Nettoyage réussi : {len(df_matches)} matchs uniques identifiés.")
print("⚙️ [2/3] Simulation chronologique et calcul du ELO_PRE (Anti-Leakage)...")

elo_dict = {}
history_dict = {}

def get_form_stats(team, n_games, current_date):
    if team not in history_dict:
        return 0.5, 1.0, 1.0, 0.0 # Form moyenne, 1 but mis, 1 but pris, 0 CS par défaut
    
    past_games = [g for g in history_dict[team] if g['date'] < current_date][-n_games:]
    if len(past_games) == 0:
        return 0.5, 1.0, 1.0, 0.0
        
    pts = sum(g['pts'] for g in past_games)
    max_pts = len(past_games) * 3
    form = pts / max_pts
    
    avg_gf = sum(g['gf'] for g in past_games) / len(past_games)
    avg_ga = sum(g['ga'] for g in past_games) / len(past_games)
    cs_ratio = sum(1 for g in past_games if g['ga'] == 0) / len(past_games)
    
    return form, avg_gf, avg_ga, cs_ratio

def get_rest_days(team, current_date):
    if team not in history_dict or len(history_dict[team]) == 0:
        return 14 # Par défaut, 2 semaines si aucune donnée historique
    last_match_date = history_dict[team][-1]['date']
    return min((current_date - last_match_date).days, 30) # Plafonné à 30 jours max

matchups = []

for idx, row in df_matches.iterrows():
    match_date = row['date']
    t1 = row['home_team']
    t2 = row['away_team']
    score1 = row['home_score']
    score2 = row['away_score']
    is_host = row['is_host_t1']
    
    # Ignorer les lignes corrompues ou les matchs futurs sans score
    if pd.isna(score1) or pd.isna(score2):
        continue
        
    # --- PHASE A : LECTURE DU TEMPS (Capture des variables avant le match) ---
    elo_pre_t1 = elo_dict.get(t1, 1500)
    elo_pre_t2 = elo_dict.get(t2, 1500)
    
    rest_t1 = get_rest_days(t1, match_date)
    rest_t2 = get_rest_days(t2, match_date)
    
    f3_t1, gf3_t1, ga3_t1, cs3_t1 = get_form_stats(t1, 3, match_date)
    f3_t2, gf3_t2, ga3_t2, cs3_t2 = get_form_stats(t2, 3, match_date)
    
    f10_t1, gf10_t1, ga10_t1, cs10_t1 = get_form_stats(t1, 10, match_date)
    f10_t2, gf10_t2, ga10_t2, cs10_t2 = get_form_stats(t2, 10, match_date)
    
    # Target (L'issue du match)
    if score1 > score2: target = '1'
    elif score1 == score2: target = 'X'
    else: target = '2'
        
    # On écrit la ligne d'entraînement. C'est ce que XGBoost va lire.
    matchups.append({
        'DATE': match_date,
        'TEAM_1': t1,
        'TEAM_2': t2,
        'TARGET_1X2': target,
        'TARGET_GOAL_DIFF': score1 - score2,
        'T1_IS_HOST': is_host,
        'ELO_PRE_DIFF': elo_pre_t1 - elo_pre_t2,
        'REST_DAYS_DIFF': rest_t1 - rest_t2,
        'FORM_3_DIFF': f3_t1 - f3_t2,
        'AVG_GOALS_3_DIFF': gf3_t1 - gf3_t2,
        'AVG_OPP_GOALS_3_DIFF': ga3_t1 - ga3_t2,
        'AVG_CLEAN_SHEET_3_DIFF': cs3_t1 - cs3_t2,
        'FORM_10_DIFF': f10_t1 - f10_t2,
        'AVG_GOALS_10_DIFF': gf10_t1 - gf10_t2,
        'AVG_OPP_GOALS_10_DIFF': ga10_t1 - ga10_t2,
        'AVG_CLEAN_SHEET_10_DIFF': cs10_t1 - cs10_t2
    })
    
    # --- PHASE B : MISE À JOUR (Intégration du résultat pour les matchs futurs) ---
    R1 = 10 ** (elo_pre_t1 / 400)
    R2 = 10 ** (elo_pre_t2 / 400)
    E1 = R1 / (R1 + R2)
    E2 = R2 / (R1 + R2)
    
    W1 = 1 if score1 > score2 else (0.5 if score1 == score2 else 0)
    W2 = 1 if score2 > score1 else (0.5 if score1 == score2 else 0)
    
    K = 40 # K-Factor Standard
    
    elo_dict[t1] = elo_pre_t1 + K * (W1 - E1)
    elo_dict[t2] = elo_pre_t2 + K * (W2 - E2)
    
    if t1 not in history_dict: history_dict[t1] = []
    history_dict[t1].append({'date': match_date, 'gf': score1, 'ga': score2, 'pts': W1 * 3 if W1 != 0.5 else 1})
    
    if t2 not in history_dict: history_dict[t2] = []
    history_dict[t2].append({'date': match_date, 'gf': score2, 'ga': score1, 'pts': W2 * 3 if W2 != 0.5 else 1})

print("💾 [3/3] Sauvegarde du dataset purgé de toute fuite...")
out_df = pd.DataFrame(matchups)
out_df.to_csv('data/football_matchups.csv', index=False)
print(f"✅ Terminé ! {len(out_df)} matchs traités. Le fichier data/football_matchups.csv est prêt pour XGBoost.")