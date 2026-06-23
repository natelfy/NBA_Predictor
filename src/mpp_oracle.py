import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. MOTEUR D'EXTRACTION DES FEATURES (LE NOUVEAU CŒUR)
# =============================================================================
def prepare_live_inference_features(team_1, team_2, current_date, history_df, elo_dict):
    """
    Reconstruit EXACTEMENT les 11 features attendues par le XGBoost
    en se basant sur le dernier état connu avant la date du match.
    """
    elo_pre_t1 = elo_dict.get(team_1, 1500)
    elo_pre_t2 = elo_dict.get(team_2, 1500)
    elo_pre_diff = elo_pre_t1 - elo_pre_t2
    
    past_t1 = history_df[(history_df['TEAM_ID'] == team_1) & (history_df['MATCH_DATE'] < current_date)].sort_values('MATCH_DATE')
    past_t2 = history_df[(history_df['TEAM_ID'] == team_2) & (history_df['MATCH_DATE'] < current_date)].sort_values('MATCH_DATE')
    
    def calculate_metrics(past_data, n_games):
        if len(past_data) == 0:
            return 0.5, 1.0, 1.0, 0.0
        recent = past_data.tail(n_games)
        
        avg_gf = recent['GOALS'].mean() if 'GOALS' in recent.columns else 1.0
        avg_ga = recent['OPP_GOALS'].mean() if 'OPP_GOALS' in recent.columns else 1.0
        
        # Calcul de la forme réelle (Victoire=3, Nul=1, Défaite=0)
        pts = np.where(recent['GOALS'] > recent['OPP_GOALS'], 3, np.where(recent['GOALS'] == recent['OPP_GOALS'], 1, 0))
        form = pts.sum() / (len(recent) * 3) if len(recent) > 0 else 0.5
        
        cs_ratio = (recent['OPP_GOALS'] == 0).sum() / len(recent) if 'OPP_GOALS' in recent.columns else 0.0
        return form, avg_gf, avg_ga, cs_ratio

    f3_t1, gf3_t1, ga3_t1, cs3_t1 = calculate_metrics(past_t1, 3)
    f3_t2, gf3_t2, ga3_t2, cs3_t2 = calculate_metrics(past_t2, 3)
    
    f10_t1, gf10_t1, ga10_t1, cs10_t1 = calculate_metrics(past_t1, 10)
    f10_t2, gf10_t2, ga10_t2, cs10_t2 = calculate_metrics(past_t2, 10)
    
    rest_t1 = (pd.to_datetime(current_date) - pd.to_datetime(past_t1['MATCH_DATE'].max())).days if len(past_t1) > 0 else 14
    rest_t2 = (pd.to_datetime(current_date) - pd.to_datetime(past_t2['MATCH_DATE'].max())).days if len(past_t2) > 0 else 14
    
    rest_t1 = min(max(rest_t1, 0), 30)
    rest_t2 = min(max(rest_t2, 0), 30)

    features = {
        'ELO_PRE_DIFF': elo_pre_diff,
        'REST_DAYS_DIFF': rest_t1 - rest_t2,
        'FORM_3_DIFF': f3_t1 - f3_t2,
        'AVG_GOALS_3_DIFF': gf3_t1 - gf3_t2,
        'AVG_OPP_GOALS_3_DIFF': ga3_t1 - ga3_t2,
        'AVG_CLEAN_SHEET_3_DIFF': cs3_t1 - cs3_t2,
        'FORM_10_DIFF': f10_t1 - f10_t2,
        'AVG_GOALS_10_DIFF': gf10_t1 - gf10_t2,
        'AVG_OPP_GOALS_10_DIFF': ga10_t1 - ga10_t2,
        'AVG_CLEAN_SHEET_10_DIFF': cs10_t1 - cs10_t2,
        'T1_IS_HOST': 0 
    }
    
    return pd.DataFrame([features])

# =============================================================================
# 2. CHARGEMENT DE L'ENVIRONNEMENT
# =============================================================================
def load_environment():
    try:
        clf = joblib.load('models/football_classifier.pkl')
        reg = joblib.load('models/football_regressor.pkl')
        with open('models/football_features.txt', 'r') as f:
            features = [line.strip() for line in f.read().split(',') if line.strip()]
        
        # On charge l'historique COMPLET (et non plus juste la dernière ligne)
        df = pd.read_csv('data/processed_football_games.csv')
        df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
        
        return clf, reg, features, df
    except Exception as e:
        print(f"❌ Erreur de chargement de l'environnement: {e}")
        return None, None, None, None

# =============================================================================
# 3. CALCUL STRATÉGIQUE (L'INFÉRENCE CORRIGÉE)
# =============================================================================
def get_mpp_strategy(team_1, team_2, current_date, mpp_odds, points_gap, is_t1_host=0):
    clf, reg, features_list, history_df = load_environment()
    if clf is None: return
    
    # A. Reconstruction du dictionnaire ELO (L'ELO_POST d'hier est l'ELO_PRE d'aujourd'hui)
    elo_dict = {}
    for team in [team_1, team_2]:
        team_history = history_df[history_df['TEAM_ID'] == team].sort_values('MATCH_DATE')
        if not team_history.empty:
            elo_dict[team] = team_history.iloc[-1]['ELO_POST']
        else:
            elo_dict[team] = 1500

    # B. Génération des variables parfaites
    X_pred = prepare_live_inference_features(team_1, team_2, current_date, history_df, elo_dict)
    X_pred['T1_IS_HOST'] = is_t1_host
    
    # C. LE GARDE-FOU (Bloquant si le Train/Serve Skew revient)
    assert X_pred.shape[1] == 11, f"🚨 Erreur Fatale : {X_pred.shape[1]} variables générées au lieu des 11 attendues par le modèle."
    
    # D. Alignement strict de l'ordre des colonnes avec le modèle d'entraînement
    try:
        X_pred = X_pred[features_list]
    except KeyError as e:
        print(f"🚨 Erreur : Variables manquantes par rapport à l'entraînement : {e}")
        return

    # E. Prédictions
    probs = clf.predict_proba(X_pred)[0] # [T2(0), NUL(1), T1(2)]
    goal_diff = reg.predict(X_pred)[0]
    
    prob_t1, prob_nul, prob_t2 = probs[2], probs[1], probs[0]
    
    ev_t1 = prob_t1 * mpp_odds['T1']
    ev_nul = prob_nul * mpp_odds['NUL']
    ev_t2 = prob_t2 * mpp_odds['T2']
    
    print("\n" + "="*50)
    print(f"🔮 ANALYSE ORACLE : {team_1.upper()} vs {team_2.upper()} (Date: {current_date})")
    print("="*50)
    print(f"📊 Probabilités : {team_1} ({prob_t1:.1%}) | Nul ({prob_nul:.1%}) | {team_2} ({prob_t2:.1%})")
    print(f"💰 Expected Value: {team_1} ({ev_t1:.1f} pts) | Nul ({ev_nul:.1f} pts) | {team_2} ({ev_t2:.1f} pts)")
    print(f"📈 Différentiel ELO calculé : {X_pred['ELO_PRE_DIFF'].values[0]:.0f} points")
    print("-" * 50)
    
    decision = ""
    score_exact = ""
    
    print(f"🎯 Contexte : {points_gap} points. Mode : ", end="")
    
    if points_gap >= 30:
        print("🛡️ LEADER (Couverture de risque)")
        choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
        best_choice = max(choices, key=choices.get)
    elif points_gap <= -40:
        print("⚔️ CHASSEUR (Recherche d'anomalie)")
        choices = {'T1': ev_t1, 'NUL': ev_nul, 'T2': ev_t2}
        best_choice = max(choices, key=choices.get)
        
        if (best_choice == 'T1' and prob_t1 < 0.15) or \
           (best_choice == 'T2' and prob_t2 < 0.15) or \
           (best_choice == 'NUL' and prob_nul < 0.15):
            print("   ⚠️ L'anomalie est trop risquée (<15%). Repli sur la probabilité.")
            prob_choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
            best_choice = max(prob_choices, key=prob_choices.get)
    else:
        print("⚖️ NEUTRE (Optimisation de la valeur)")
        valid_choices = {k: v for k, v in zip(['T1', 'NUL', 'T2'], [ev_t1, ev_nul, ev_t2]) if [prob_t1, prob_nul, prob_t2][['T1', 'NUL', 'T2'].index(k)] >= 0.28}
        
        if not valid_choices:
            choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
            best_choice = max(choices, key=choices.get)
        else:
            best_choice = max(valid_choices, key=valid_choices.get)

    abs_diff = abs(goal_diff)
    
    if best_choice == 'T1':
        if abs_diff < 1.0: score_exact = "1 - 0 (ou 2 - 1)"
        elif abs_diff < 2.0: score_exact = "2 - 0"
        else: score_exact = "3 - 0"
        winner = team_1
    elif best_choice == 'T2':
        if abs_diff < 1.0: score_exact = "0 - 1 (ou 1 - 2)"
        elif abs_diff < 2.0: score_exact = "0 - 2"
        else: score_exact = "0 - 3"
        winner = team_2
    else:
        if abs_diff < 0.5: score_exact = "0 - 0"
        else: score_exact = "1 - 1"
        winner = "MATCH NUL"
        
    print(f"\n✅ RECOMMANDATION STRICTE : Jouer {winner.upper()}")
    print(f"⚽ SCORE CONSEILLÉ : {score_exact}")
    print("="*50 + "\n")

# =============================================================================
# ZONE DE TEST (LA VÉRIFICATION)
# =============================================================================
if __name__ == "__main__":
    today_date = "2026-06-23"
    points_retard = -560  # Ton écart actuel
    
    # Dictionnaire de tes matchs du jour avec les cotes MPP
    matchs_du_jour = [
        {"t1": "Portugal", "t2": "Uzbekistan", "cotes": {'T1': 24, 'NUL': 154, 'T2': 173}},
        {"t1": "England", "t2": "Ghana", "cotes": {'T1': 44, 'NUL': 127, 'T2': 162}},
        {"t1": "Panama", "t2": "Croatia", "cotes": {'T1': 165, 'NUL': 123, 'T2': 36}},
        {"t1": "Colombia", "t2": "DR Congo", "cotes": {'T1': 77, 'NUL': 111, 'T2': 123}}
    ]
    
    for match in matchs_du_jour:
        get_mpp_strategy(
            team_1=match["t1"], 
            team_2=match["t2"], 
            current_date=today_date,
            mpp_odds=match["cotes"], 
            points_gap=points_retard, 
            is_t1_host=0
        )