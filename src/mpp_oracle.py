import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CHARGEMENT DE L'ENVIRONNEMENT
# =============================================================================

def load_environment():
    try:
        clf = joblib.load('models/football_classifier.pkl')
        reg = joblib.load('models/football_regressor.pkl')
        with open('models/football_features.txt', 'r') as f:
            features = f.read().split(',')
        
        # On charge le dernier état connu des équipes
        df = pd.read_csv('data/processed_football_games.csv')
        df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
        latest_stats = df.sort_values('MATCH_DATE').groupby('TEAM_ID').last().reset_index()
        
        return clf, reg, features, latest_stats
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None, None, None

# =============================================================================
# 2. CALCUL STRATÉGIQUE (THÉORIE DES JEUX)
# =============================================================================

def get_mpp_strategy(team_1, team_2, mpp_odds, points_gap, is_t1_host=0):
    """
    team_1, team_2 : Noms des équipes
    mpp_odds : Dict avec les points offerts par MPP {'T1': 14, 'NUL': 45, 'T2': 82}
    points_gap : Ton retard (-) ou avance (+) sur le 1er/poursuivant. Ex: -50 (tu as 50 pts de retard)
    """
    clf, reg, features, stats = load_environment()
    if clf is None: return
    
    # 1. Récupérer les stats des deux équipes
    t1_stats = stats[stats['TEAM_ID'] == team_1]
    t2_stats = stats[stats['TEAM_ID'] == team_2]
    
    if t1_stats.empty or t2_stats.empty:
        print(f"❌ Équipe non trouvée dans la base de données.")
        return

    # 2. Construire la ligne de prédiction (Calcul des _DIFF)
    pred_data = {}
    for feat in features:
        if feat == 'T1_IS_HOST':
            pred_data[feat] = [is_t1_host]
        else:
            base_feat = feat.replace('_DIFF', '')
            val_t1 = t1_stats[base_feat].values[0]
            val_t2 = t2_stats[base_feat].values[0]
            pred_data[feat] = [val_t1 - val_t2]
            
    X_pred = pd.DataFrame(pred_data)[features]
    
    # 3. Prédictions brutes
    probs = clf.predict_proba(X_pred)[0] # [Prob T2 (0), Prob NUL (1), Prob T1 (2)]
    goal_diff = reg.predict(X_pred)[0]   # Ecart de buts (Positif = T1, Négatif = T2)
    
    prob_t1, prob_nul, prob_t2 = probs[2], probs[1], probs[0]
    
    # 4. Calcul de l'Expected Value (EV = Probabilité * Points MPP)
    ev_t1 = prob_t1 * mpp_odds['T1']
    ev_nul = prob_nul * mpp_odds['NUL']
    ev_t2 = prob_t2 * mpp_odds['T2']
    
    print("\n" + "="*50)
    print(f"🔮 ANALYSE ORACLE : {team_1.upper()} vs {team_2.upper()}")
    print("="*50)
    print(f"📊 Probabilités : {team_1} ({prob_t1:.1%}) | Nul ({prob_nul:.1%}) | {team_2} ({prob_t2:.1%})")
    print(f"💰 Expected Value: {team_1} ({ev_t1:.1f} pts) | Nul ({ev_nul:.1f} pts) | {team_2} ({ev_t2:.1f} pts)")
    print(f"📈 Différentiel ELO: {pred_data['ELO_PRE_DIFF'][0]:.0f} points")
    print("-" * 50)
    
    # 5. MOTEUR DE DÉCISION (Le coeur du système)
    decision = ""
    score_exact = ""
    
    print(f"🎯 Contexte de ligue : {points_gap} points. Mode activé : ", end="")
    
    if points_gap >= 30:
        # MODE LEADER : Couverture. On minimise le risque de "Bulle". On prend la plus haute proba.
        print("🛡️ LEADER (Couverture de risque)")
        choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
        best_choice = max(choices, key=choices.get)
        
    elif points_gap <= -40:
        # MODE CHASSEUR : Contre-courant. On veut l'EV maximum absolue pour créer l'anomalie.
        print("⚔️ CHASSEUR (Recherche d'anomalie)")
        choices = {'T1': ev_t1, 'NUL': ev_nul, 'T2': ev_t2}
        best_choice = max(choices, key=choices.get)
        
        # Sécurité anti-suicide : Si l'EV est haute mais la proba est < 15%, c'est trop risqué.
        if (best_choice == 'T1' and prob_t1 < 0.15) or \
           (best_choice == 'T2' and prob_t2 < 0.15) or \
           (best_choice == 'NUL' and prob_nul < 0.15):
            print("   ⚠️ L'anomalie est trop risquée (<15%). Repli sur la probabilité.")
            prob_choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
            best_choice = max(prob_choices, key=prob_choices.get)
            
    else:
        # MODE NEUTRE : Optimisation équilibrée. EV forte mais proba décente (> 28%).
        print("⚖️ NEUTRE (Optimisation de la valeur)")
        valid_choices = {}
        if prob_t1 >= 0.28: valid_choices['T1'] = ev_t1
        if prob_nul >= 0.28: valid_choices['NUL'] = ev_nul
        if prob_t2 >= 0.28: valid_choices['T2'] = ev_t2
        
        if not valid_choices: # Si aucun choix n'est > 28% (très rare), on prend la meilleure proba.
            choices = {'T1': prob_t1, 'NUL': prob_nul, 'T2': prob_t2}
            best_choice = max(choices, key=choices.get)
        else:
            best_choice = max(valid_choices, key=valid_choices.get)

    # 6. Déduction du Score Exact (Basé sur le choix stratégique et la prédiction de buts)
    abs_diff = abs(goal_diff)
    
    if best_choice == 'T1':
        if abs_diff < 1.0: score_exact = "1 - 0" or "2 - 1"
        elif abs_diff < 2.0: score_exact = "2 - 0"
        else: score_exact = "3 - 0"
        winner = team_1
        
    elif best_choice == 'T2':
        if abs_diff < 1.0: score_exact = "0 - 1" or "1 - 2"
        elif abs_diff < 2.0: score_exact = "0 - 2"
        else: score_exact = "0 - 3"
        winner = team_2
        
    else:
        if abs_diff < 0.5: score_exact = "0 - 0"
        else: score_exact = "1 - 1" # Statistique la plus probable d'un nul
        winner = "MATCH NUL"
        
    print(f"\n✅ RECOMMANDATION STRICTE : Jouer {winner.upper()}")
    print(f"⚽ SCORE CONSEILLÉ : {score_exact}")
    print("="*50 + "\n")

# =============================================================================
# ZONE DE TEST (À MODIFIER SELON TA LIGUE)
# =============================================================================
if __name__ == "__main__":
    
    # Scénario de test fictif : France vs Maroc. 
    # La France est favorite (MPP donne 14 points), le Maroc outsider (MPP donne 85 points).
    # Le Nul rapporte 42 points.
    
    mpp_cotes = {'T1': 14, 'NUL': 42, 'T2': 85}
    
    # Test 1 : Tu es premier de ta ligue (+45 points d'avance)
    get_mpp_strategy(
        team_1="France", 
        team_2="Morocco", 
        mpp_odds=mpp_cotes, 
        points_gap=45, 
        is_t1_host=0
    )
    
    # Test 2 : Tu es dernier de ta ligue (-60 points de retard sur le leader)
    get_mpp_strategy(
        team_1="France", 
        team_2="Morocco", 
        mpp_odds=mpp_cotes, 
        points_gap=-60, 
        is_t1_host=0
    )