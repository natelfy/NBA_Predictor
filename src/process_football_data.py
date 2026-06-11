import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. CALCUL ELO (AVEC RECENCY BIAS MULTIPLIER)
# =============================================================================

def calculate_football_elo(df, base_home_advantage=60):
    print("📊 Calcul de l'ELO Historique (avec Time Decay post-2023)...")
    df = df.sort_values(['MATCH_DATE', 'MATCH_ID']).copy()
    
    elo_ratings = {team: 1500 for team in df['TEAM_ID'].unique()}
    
    df['ELO_PRE'] = 0.0
    df['ELO_POST'] = 0.0
    
    k_factors = {
        'World Cup': 60,
        'Continental Cup': 50,
        'Qualifier': 40,
        'Nations League': 30,
        'Friendly': 20,
        'Other': 20
    }
    
    matches = df.groupby('MATCH_ID').agg({
        'TEAM_ID': list,
        'GOALS': list,
        'IS_HOST': list,
        'TOURNAMENT_CATEGORY': 'first',
        'MATCH_DATE': 'first'
    }).reset_index()
    
    for _, match in matches.iterrows():
        if len(match['TEAM_ID']) != 2:
            continue
            
        team_a, team_b = match['TEAM_ID'][0], match['TEAM_ID'][1]
        goals_a, goals_b = match['GOALS'][0], match['GOALS'][1]
        host_a, host_b = match['IS_HOST'][0], match['IS_HOST'][1]
        year = pd.to_datetime(match['MATCH_DATE']).year
        
        elo_a, elo_b = elo_ratings[team_a], elo_ratings[team_b]
        
        elo_a_adj = elo_a + (base_home_advantage if host_a == 1 else 0)
        elo_b_adj = elo_b + (base_home_advantage if host_b == 1 else 0)
        
        expected_a = 1 / (1 + 10 ** ((elo_b_adj - elo_a_adj) / 400))
        
        if goals_a > goals_b: result_a = 1.0
        elif goals_a == goals_b: result_a = 0.5
        else: result_a = 0.0
            
        goal_diff = abs(goals_a - goals_b)
        mov_mult = 1.0 if goal_diff <= 1 else 1.0 + 0.75 * np.log(goal_diff)
            
        k = k_factors.get(match['TOURNAMENT_CATEGORY'], 20)
        
        # LE SECRET EST ICI : Le multiplicateur de récence
        # Les matchs joués depuis 2023 ont un impact 1.5x plus fort sur la note
        recency_mult = 1.5 if year >= 2023 else 1.0
        
        change = k * mov_mult * recency_mult * (result_a - expected_a)
        
        mask_a = (df['MATCH_ID'] == match['MATCH_ID']) & (df['TEAM_ID'] == team_a)
        mask_b = (df['MATCH_ID'] == match['MATCH_ID']) & (df['TEAM_ID'] == team_b)
        
        df.loc[mask_a, 'ELO_PRE'] = elo_a
        df.loc[mask_a, 'ELO_POST'] = elo_a + change
        
        df.loc[mask_b, 'ELO_PRE'] = elo_b
        df.loc[mask_b, 'ELO_POST'] = elo_b - change
        
        elo_ratings[team_a] += change
        elo_ratings[team_b] -= change

    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def add_football_features(df):
    print("🔧 Ingénierie des variables de performance...")
    df = df.sort_values(['TEAM_ID', 'MATCH_DATE']).copy()
    
    df['CLEAN_SHEET'] = (df['OPP_GOALS'] == 0).astype(int)
    
    conditions = [df['GOALS'] > df['OPP_GOALS'], df['GOALS'] == df['OPP_GOALS'], df['GOALS'] < df['OPP_GOALS']]
    df['MATCH_POINTS'] = np.select(conditions, [3, 1, 0], default=0)
    df['GOAL_DIFF_MATCH'] = df['GOALS'] - df['OPP_GOALS']
    
    df['PREV_MATCH_DATE'] = df.groupby('TEAM_ID')['MATCH_DATE'].shift(1)
    df['REST_DAYS'] = (pd.to_datetime(df['MATCH_DATE']) - pd.to_datetime(df['PREV_MATCH_DATE'])).dt.days
    df['REST_DAYS'] = df['REST_DAYS'].fillna(30).clip(0, 30)
    
    return df.drop(columns=['PREV_MATCH_DATE'], errors='ignore')

# =============================================================================
# 3. MOYENNES EXPONENTIELLES (L'ARME TACTIQUE)
# =============================================================================

def add_rolling_averages(df, windows=[3, 10]):
    print("📈 Calcul des dynamiques (Moyennes Exponentielles - EMA)...")
    df = df.sort_values(['TEAM_ID', 'MATCH_DATE']).copy()
    
    for w in windows:
        # Remplacement de rolling.mean() par ewm.mean()
        # Cela donne un poids massif au dernier match joué
        df[f'FORM_{w}'] = df.groupby('TEAM_ID')['MATCH_POINTS'].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        df[f'AVG_GOALS_{w}'] = df.groupby('TEAM_ID')['GOALS'].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        df[f'AVG_OPP_GOALS_{w}'] = df.groupby('TEAM_ID')['OPP_GOALS'].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        df[f'AVG_CLEAN_SHEET_{w}'] = df.groupby('TEAM_ID')['CLEAN_SHEET'].transform(
            lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
        )
        
    return df

# =============================================================================
# 4. CRÉATION DES MATCHUPS (FORMAT MACHINE LEARNING)
# =============================================================================

def create_matchups(df):
    print("⚽ Création du dataset d'entraînement final...")
    
    # Séparation T1 / T2 arbitraire par index pair/impair
    df = df.sort_values(['MATCH_DATE', 'MATCH_ID', 'TEAM_ID']).copy()
    df['TEAM_INDEX'] = df.groupby('MATCH_ID').cumcount()
    
    t1 = df[df['TEAM_INDEX'] == 0].copy()
    t2 = df[df['TEAM_INDEX'] == 1].copy()
    
    id_cols = ['MATCH_ID', 'MATCH_DATE', 'TOURNAMENT_CATEGORY', 'IS_HOST']
    feature_cols = [c for c in t1.columns if c.startswith(('ELO_PRE', 'REST_DAYS', 'FORM', 'AVG_'))]
    
    t1_feat = t1[id_cols + feature_cols + ['TEAM_ID', 'GOALS']].rename(
        columns={c: f'T1_{c}' for c in feature_cols + ['TEAM_ID', 'GOALS']}
    )
    # L'hôte est toujours affecté via T1 ou T2, on garde l'info globale si T1 est hôte
    t1_feat['T1_IS_HOST'] = t1['IS_HOST']
    t1_feat = t1_feat.drop(columns=['IS_HOST'])

    t2_feat = t2[['MATCH_ID'] + feature_cols + ['TEAM_ID', 'GOALS', 'IS_HOST']].rename(
        columns={c: f'T2_{c}' for c in feature_cols + ['TEAM_ID', 'GOALS', 'IS_HOST']}
    )
    
    matchups = pd.merge(t1_feat, t2_feat, on='MATCH_ID', how='inner')
    
    # --- DIFFÉRENTIELS (LES VARIABLES CLÉS) ---
    for feat in feature_cols:
        matchups[f'{feat}_DIFF'] = matchups[f'T1_{feat}'] - matchups[f'T2_{feat}']
        
    # --- CIBLES (TARGETS) ---
    matchups['TARGET_GOALS_T1'] = matchups['T1_GOALS']
    matchups['TARGET_GOALS_T2'] = matchups['T2_GOALS']
    matchups['TARGET_GOAL_DIFF'] = matchups['T1_GOALS'] - matchups['T2_GOALS']
    
    # 0 = T2 Win, 1 = Draw, 2 = T1 Win
    matchups['TARGET_1X2'] = np.select(
        [matchups['TARGET_GOAL_DIFF'] > 0, matchups['TARGET_GOAL_DIFF'] == 0, matchups['TARGET_GOAL_DIFF'] < 0],
        [2, 1, 0]
    )
    
    # Filtrer depuis 2018 pour l'entraînement (Le foot d'avant n'est plus représentatif tactiquement)
    matchups = matchups[pd.to_datetime(matchups['MATCH_DATE']).dt.year >= 2018]
    
    # Nettoyage
    cols_to_drop = [c for c in matchups.columns if c.startswith(('T1_AVG', 'T2_AVG', 'T1_FORM', 'T2_FORM'))]
    matchups = matchups.drop(columns=cols_to_drop + ['T1_GOALS', 'T2_GOALS'])
    
    return matchups

# =============================================================================
# 5. EXECUTION PIPELINE
# =============================================================================

def process_pipeline():
    raw_path = 'data/raw_international_games.csv'
    
    print("=" * 60)
    print("🧠 FOOTBALL ORACLE - PIPELINE DE TRAITEMENT")
    print("=" * 60)
    
    df = pd.read_csv(raw_path)
    df = calculate_football_elo(df)
    df = add_football_features(df)
    df = add_rolling_averages(df)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/processed_football_games.csv', index=False)
    
    matchups = create_matchups(df)
    matchups = matchups.dropna()
    matchups.to_csv('data/football_matchups.csv', index=False)
    
    print("\n" + "=" * 60)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print(f"📊 Données d'entraînement : {len(matchups)} matchs (Depuis 2018)")
    
    # Vérification Sanity Check (Top 5 ELO actuel)
    latest = df.sort_values('MATCH_DATE').groupby('TEAM_ID').last()
    top_5 = latest.sort_values('ELO_POST', ascending=False).head(5)
    print(f"\n🏆 Top 5 ELO actuel (Sanity Check):")
    for team, row in top_5.iterrows():
        print(f"   {team}: {row['ELO_POST']:.0f}")

if __name__ == "__main__":
    process_pipeline()