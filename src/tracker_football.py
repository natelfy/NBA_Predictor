import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import log_loss, accuracy_score
import os
import sys

# Connecter le moteur d'extraction pour recalculer les conditions exactes pré-match
from mpp_oracle import prepare_live_inference_features

def run_world_cup_tracking():
    print("=" * 60)
    print("📡 RADAR DE PERFORMANCE (COUPE DU MONDE 2026)")
    print("=" * 60)

    results_path = 'data/wc_2026_results.csv'
    if not os.path.exists(results_path):
        print("❌ Fichier de résultats de la Coupe du Monde introuvable.")
        return

    wc_df = pd.read_csv(results_path, names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
    history_df = pd.read_csv('data/processed_football_games.csv')
    
    clf = joblib.load('models/football_classifier.pkl')
    with open('models/football_features.txt', 'r') as f:
        features_list = [line.strip() for line in f.read().split(',') if line.strip()]

    predictions_log = []
    
    print(f"🔍 Analyse de {len(wc_df)} matchs réels joués...")
    
    # Pour chaque match réel du Mondial, on va voir ce que le modèle aurait dit AVANT le coup d'envoi
    for idx, row in wc_df.iterrows():
        match_date = str(row['DATE'])
        t1, t2 = row['HOME'], row['AWAY']
        hg, ag = row['HG'], row['AG']
        
        # Détermination de la cible (0=T2, 1=NUL, 2=T1)
        target = 2 if hg > ag else (1 if hg == ag else 0)
        
        # Reconstruction ELO
        elo_dict = {}
        for t in [t1, t2]:
            team_hist = history_df[(history_df['TEAM_ID'] == t) & (history_df['MATCH_DATE'] < match_date)].sort_values('MATCH_DATE')
            elo_dict[t] = team_hist.iloc[-1]['ELO_POST'] if not team_hist.empty else 1500
            
        # Extraction Features
        X_pred = prepare_live_inference_features(t1, t2, match_date, history_df, elo_dict)
        X_pred['T1_IS_HOST'] = 0 # Par défaut
        X_pred = X_pred[features_list]
        
        # Prédiction
        probs = clf.predict_proba(X_pred)[0]
        
        predictions_log.append({
            'DATE': match_date,
            'MATCH': f"{t1} vs {t2}",
            'PROB_T1': probs[2],
            'PROB_NUL': probs[1],
            'PROB_T2': probs[0],
            'PREDICTED_CLASS': np.argmax(probs),
            'ACTUAL_CLASS': target
        })

    eval_df = pd.DataFrame(predictions_log)
    
    # Métriques de la vraie compétition
    acc = accuracy_score(eval_df['ACTUAL_CLASS'], eval_df['PREDICTED_CLASS'])
    
    print(f"\n📊 Bilan des performances en direct sur la Coupe du Monde :")
    print(f"   ✅ Accuracy : {acc:.1%}")
    print("\n📉 Détail des erreurs / surprises :")
    
    erreurs = eval_df[eval_df['PREDICTED_CLASS'] != eval_df['ACTUAL_CLASS']]
    for _, err in erreurs.tail(5).iterrows():
        print(f"   ⚠️ {err['MATCH']} | Attendu: Classe {err['PREDICTED_CLASS']} | Réel: Classe {err['ACTUAL_CLASS']}")

    eval_df.to_csv('data/mpp_live_tracking.csv', index=False)
    print("\n💾 Rapport détaillé sauvegardé dans data/mpp_live_tracking.csv")
    print("=" * 60)

if __name__ == "__main__":
    run_world_cup_tracking()