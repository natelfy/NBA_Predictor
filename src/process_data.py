import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. CALCUL ELO
# =============================================================================

def calculate_elo(df, k_factor=20, home_advantage=100):
    """
    Calcul ELO CORRECT qui prend en compte:
    - La force de l'adversaire (probabilité attendue)
    - L'avantage du terrain
    - La marge de victoire (optionnel mais utile)
    
    Formule: new_elo = old_elo + K * (result - expected)
    Où expected = 1 / (1 + 10^((elo_adversaire - elo_equipe) / 400))
    """
    print("📊 Calcul de l'ELO (méthode corrigée)...")
    
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).copy()
    
    # Initialisation des ELO
    elo_ratings = {team: 1500 for team in df['TEAM_ID'].unique()}
    
    # Colonnes de sortie
    df['ELO_PRE'] = 0.0
    df['ELO_POST'] = 0.0
    df['ELO_DIFF'] = 0.0  # Différence vs adversaire
    
    # Grouper par match (chaque match = 2 lignes)
    games = df.groupby('GAME_ID').agg({
        'TEAM_ID': list,
        'MATCHUP': list,
        'WL': list,
        'PLUS_MINUS': 'first',
        'GAME_DATE': 'first'
    }).reset_index()
    
    for _, game in games.iterrows():
        game_id = game['GAME_ID']
        teams = game['TEAM_ID']
        matchups = game['MATCHUP']
        results = game['WL']
        margin = abs(game['PLUS_MINUS']) if pd.notna(game['PLUS_MINUS']) else 0
        
        if len(teams) != 2:
            continue
        
        team_a, team_b = teams[0], teams[1]
        elo_a, elo_b = elo_ratings[team_a], elo_ratings[team_b]
        
        # Qui est à domicile? (pas de '@' = domicile)
        is_a_home = '@' not in str(matchups[0])
        
        # Ajustement domicile pour le calcul de la probabilité
        elo_a_adj = elo_a + (home_advantage if is_a_home else 0)
        elo_b_adj = elo_b + (home_advantage if not is_a_home else 0)
        
        # Probabilité attendue (formule ELO standard)
        expected_a = 1 / (1 + 10 ** ((elo_b_adj - elo_a_adj) / 400))
        
        # Résultat réel
        result_a = 1.0 if results[0] == 'W' else 0.0
        
        # Multiplicateur de marge (diminishing returns)
        # Plus la marge est grande, plus l'impact, mais avec un cap
        mov_mult = np.log(margin + 1) * 0.5 + 1 if margin > 0 else 1.0
        mov_mult = min(mov_mult, 2.5)  # Cap à 2.5x
        
        # Calcul des changements
        change = k_factor * mov_mult * (result_a - expected_a)
        
        # Mise à jour dans le DataFrame
        mask_a = (df['GAME_ID'] == game_id) & (df['TEAM_ID'] == team_a)
        mask_b = (df['GAME_ID'] == game_id) & (df['TEAM_ID'] == team_b)
        
        df.loc[mask_a, 'ELO_PRE'] = elo_a
        df.loc[mask_a, 'ELO_POST'] = elo_a + change
        df.loc[mask_a, 'ELO_DIFF'] = elo_a - elo_b  # Diff vs adversaire (sans home adv)
        
        df.loc[mask_b, 'ELO_PRE'] = elo_b
        df.loc[mask_b, 'ELO_POST'] = elo_b - change
        df.loc[mask_b, 'ELO_DIFF'] = elo_b - elo_a
        
        # Mise à jour des ratings pour les prochains matchs
        elo_ratings[team_a] += change
        elo_ratings[team_b] -= change
    
    print(f"   ✅ ELO calculé pour {len(games)} matchs")
    print(f"   📈 ELO max: {df['ELO_POST'].max():.0f}, min: {df['ELO_POST'].min():.0f}")
    
    return df


# =============================================================================
# 2. FEATURES AVANCÉES
# =============================================================================

def add_advanced_features(df):
    """
    Ajoute les features avancées manquantes:
    - Four Factors (EFG%, TOV%, OREB%, FT Rate)
    - Ratings (ORtg, DRtg, Net Rating)
    - Contexte (B2B, Streak, Form)
    """
    print("🔧 Ajout des features avancées...")
    
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).copy()
    
    # --- FOUR FACTORS ---
    # eFG% = (FGM + 0.5 * FG3M) / FGA
    df['EFG_PCT'] = np.where(
        df['FGA'] > 0,
        (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'],
        0
    )
    
    # TOV% = TOV / (FGA + 0.44*FTA + TOV)
    df['POSS_EST'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    df['TOV_PCT'] = np.where(df['POSS_EST'] > 0, df['TOV'] / df['POSS_EST'], 0)
    
    # OREB% = OREB / (OREB + DREB)
    total_reb = df['OREB'] + df['DREB']
    df['OREB_PCT'] = np.where(total_reb > 0, df['OREB'] / total_reb, 0)
    
    # FT Rate = FTM / FGA
    df['FT_RATE'] = np.where(df['FGA'] > 0, df['FTM'] / df['FGA'], 0)
    
    # --- RATINGS ---
    # ORtg = PTS / POSS * 100
    df['ORTG'] = np.where(df['POSS_EST'] > 0, (df['PTS'] / df['POSS_EST']) * 100, 0)
    
    # Points adverses (approximation via PLUS_MINUS)
    df['OPP_PTS'] = df['PTS'] - df['PLUS_MINUS']
    df['DRTG'] = np.where(df['POSS_EST'] > 0, (df['OPP_PTS'] / df['POSS_EST']) * 100, 0)
    
    # Net Rating
    df['NET_RTG'] = df['ORTG'] - df['DRTG']
    
    # --- CONTEXTE ---
    # WIN (si pas déjà présent)
    if 'WIN' not in df.columns:
        df['WIN'] = (df['WL'] == 'W').astype(int)
    
    # IS_HOME
    if 'IS_HOME' not in df.columns:
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in str(x) else 1)
    
    # REST_DAYS
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['REST_DAYS'] = df['REST_DAYS'].fillna(3).clip(0, 7)
    
    # IS_B2B (Back-to-Back)
    df['IS_B2B'] = (df['REST_DAYS'] <= 1).astype(int)
    
    # STREAK (série en cours) - calculé AVANT le match actuel
    def calc_streak(wins):
        streak = []
        current = 0
        for win in wins:
            streak.append(current)  # Streak AVANT ce match
            if win == 1:
                current = current + 1 if current >= 0 else 1
            else:
                current = current - 1 if current <= 0 else -1
        return streak
    
    df['STREAK'] = df.groupby('TEAM_ID')['WIN'].transform(
        lambda x: pd.Series(calc_streak(x.values), index=x.index)
    )
    
    # FORM (% victoires sur 5 et 10 derniers matchs)
    for window in [5, 10]:
        df[f'FORM_{window}'] = df.groupby('TEAM_ID')['WIN'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Nettoyage
    df = df.drop(columns=['PREV_GAME_DATE'], errors='ignore')
    
    print(f"   ✅ {len(df.columns)} colonnes après feature engineering")
    
    return df


# =============================================================================
# 3. MOYENNES GLISSANTES
# =============================================================================

def add_rolling_averages(df, windows=[10]):
    """
    Calcule les moyennes glissantes AVANT chaque match (shift=1).
    C'est crucial pour éviter le data leakage!
    """
    print("📈 Calcul des moyennes glissantes...")
    
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).copy()
    
    # Colonnes à moyenner
    cols_to_roll = [
        'PTS', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE',
        'ORTG', 'DRTG', 'NET_RTG', 'PLUS_MINUS'
    ]
    
    # Garder seulement les colonnes qui existent
    cols_to_roll = [c for c in cols_to_roll if c in df.columns]
    
    for col in cols_to_roll:
        for window in windows:
            col_name = f'AVG_{col}_{window}'
            # shift(1) = exclure le match actuel (TRÈS IMPORTANT!)
            df[col_name] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).mean()
            )
    
    print(f"   ✅ Moyennes calculées pour {len(cols_to_roll)} colonnes")
    
    return df


# =============================================================================
# 4. CRÉATION DES MATCHUPS
# =============================================================================

def create_matchups(df):
    """
    Transforme les données en MATCHUPS (1 ligne = 1 match avec HOME vs AWAY).
    
    C'est LA correction la plus importante!
    Avant: on prédit "cette équipe gagne-t-elle?" sans savoir contre qui
    Après: on prédit "HOME bat-il AWAY?" avec les stats des DEUX équipes
    """
    print("🏀 Création du dataset de matchups...")
    
    df = df.sort_values('GAME_DATE').copy()
    
    # Séparer home et away
    home = df[df['IS_HOME'] == 1].copy()
    away = df[df['IS_HOME'] == 0].copy()
    
    print(f"   Matchs domicile: {len(home)}, extérieur: {len(away)}")
    
    # Colonnes d'identification (à garder une seule fois)
    id_cols = ['GAME_ID', 'GAME_DATE']
    
    # Colonnes à exclure du préfixage
    exclude = ['GAME_ID', 'GAME_DATE', 'IS_HOME', 'MATCHUP', 'WL', 'SEASON_ID', 'SEASON_TYPE']
    
    # Colonnes features
    feature_cols = [c for c in home.columns if c not in exclude]
    
    # Préfixer HOME_
    home_renamed = home[id_cols + feature_cols].copy()
    home_renamed.columns = id_cols + [f'HOME_{c}' for c in feature_cols]
    
    # Préfixer AWAY_
    away_renamed = away[['GAME_ID'] + feature_cols].copy()
    away_renamed.columns = ['GAME_ID'] + [f'AWAY_{c}' for c in feature_cols]
    
    # Merger sur GAME_ID
    matchups = home_renamed.merge(away_renamed, on='GAME_ID', how='inner')
    
    print(f"   Matchups créés: {len(matchups)}")
    
    # --- FEATURES DIFFÉRENTIELLES (les plus prédictives!) ---
    print("   Calcul des features différentielles...")
    
    # Liste des features pour lesquelles on calcule la différence
    diff_features = [
        'ELO_PRE', 'REST_DAYS', 'STREAK', 'FORM_5', 'FORM_10',
        'AVG_PTS_10', 'AVG_EFG_PCT_10', 'AVG_TOV_PCT_10',
        'AVG_OREB_PCT_10', 'AVG_FT_RATE_10', 'AVG_NET_RTG_10'
    ]
    
    for feat in diff_features:
        home_col = f'HOME_{feat}'
        away_col = f'AWAY_{feat}'
        
        if home_col in matchups.columns and away_col in matchups.columns:
            # Nom simplifié pour la différence
            if feat.startswith('AVG_'):
                diff_name = feat.replace('AVG_', '').replace('_10', '') + '_DIFF'
            else:
                diff_name = f'{feat}_DIFF'
            
            matchups[diff_name] = matchups[home_col] - matchups[away_col]
    
    # Target: victoire domicile
    matchups['HOME_WIN'] = matchups['HOME_WIN'].astype(int)
    
    print(f"   ✅ Dataset final: {len(matchups)} matchs, {len(matchups.columns)} colonnes")
    
    return matchups


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def process_data():
    """Pipeline complet de traitement des données."""
    
    print("=" * 60)
    print("🏀 NBA ORACLE - TRAITEMENT DES DONNÉES")
    print("=" * 60)
    
    # --- Chargement ---
    data_path = 'data/raw_games.csv'
    if not os.path.exists(data_path):
        print(f"❌ Fichier non trouvé: {data_path}")
        print("   Lance d'abord: python src/get_data.py")
        return
    
    print(f"\n📂 Chargement de {data_path}...")
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    print(f"   {len(df)} lignes chargées")
    
    # --- Pipeline ---
    print("\n" + "-" * 40)
    
    # 1. ELO (corrigé)
    df = calculate_elo(df)
    
    # 2. Features avancées
    df = add_advanced_features(df)
    
    # 3. Moyennes glissantes
    df = add_rolling_averages(df, windows=[10])
    
    # 4. Sauvegarde données par équipe (pour l'app)
    print("\n💾 Sauvegarde des données processées...")
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/processed_games.csv', index=False)
    print(f"   ✅ data/processed_games.csv ({len(df)} lignes)")
    
    # 5. Création des matchups (pour l'entraînement)
    matchups = create_matchups(df)
    
    # Nettoyage des NaN
    initial = len(matchups)
    matchups = matchups.dropna()
    print(f"   Nettoyage NaN: {initial} → {len(matchups)} matchups")
    
    matchups.to_csv('data/matchups.csv', index=False)
    print(f"   ✅ data/matchups.csv ({len(matchups)} matchups)")
    
    # --- Résumé ---
    print("\n" + "=" * 60)
    print("✅ TRAITEMENT TERMINÉ")
    print("=" * 60)
    print(f"   📊 Période: {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()}")
    print(f"   🏀 {len(matchups)} matchups prêts pour l'entraînement")
    print(f"   📈 % victoires domicile: {matchups['HOME_WIN'].mean():.1%}")
    
    # Top 5 ELO actuels
    latest_elo = df.sort_values('GAME_DATE').groupby('TEAM_ID').last()[['TEAM_NAME', 'ELO_POST']]
    latest_elo = latest_elo.sort_values('ELO_POST', ascending=False).head(5)
    print(f"\n   🏆 Top 5 ELO actuels:")
    for _, row in latest_elo.iterrows():
        print(f"      {row['TEAM_NAME']}: {row['ELO_POST']:.0f}")


if __name__ == "__main__":
    process_data()