import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os

def train_models():
    print("=" * 60)
    print("🧠 ENTRAÎNEMENT AVANCÉ DES MODÈLES (V2 - POIDS & TUNING)")
    print("=" * 60)

    # 1. Chargement des données
    data_path = 'data/football_matchups.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ {data_path} introuvable.")
    
    df = pd.read_csv(data_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    features = [c for c in df.columns if (c.endswith('_DIFF') and not c.startswith('TARGET_')) or c == 'T1_IS_HOST']
    X = df[features]
    
    target_map = {'2': 0, 'X': 1, '1': 2}
    y_class = df['TARGET_1X2'].astype(str).map(target_map)
    y_reg = df['TARGET_GOAL_DIFF']
    
    df = df.sort_values('DATE')
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    
    # --- LA CORRECTION DU FINDING C4 : RÉÉQUILIBRAGE DES NULS ---
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_class_train)

    # =========================================================================
    # MODÈLE 1 : CLASSIFICATION AVEC EARLY STOPPING (API RÉCENTE)
    # =========================================================================
    print("\n[1/2] Entraînement Classification (Avec Équilibrage des Nuls)...")
    
    clf_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 20  # <-- Déplacé ici pour la compatibilité
    }
    
    clf = xgb.XGBClassifier(**clf_params)
    
    clf.fit(
        X_train, y_class_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_class_train), (X_test, y_class_test)],
        verbose=False
    )
    
    class_preds = clf.predict(X_test)
    class_probs = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_class_test, class_preds)
    ll = log_loss(y_class_test, class_probs)
    
    brier_scores = [brier_score_loss((y_class_test == k).astype(int), class_probs[:, k]) for k in range(3)]
    avg_brier = np.mean(brier_scores)
    
    p_t2, p_nul, p_t1 = (y_class_test == 0).mean(), (y_class_test == 1).mean(), (y_class_test == 2).mean()
    ll_baseline_naive = log_loss(y_class_test, np.tile([p_t2, p_nul, p_t1], (len(y_class_test), 1)))

    print(f"   🎯 Log Loss : {ll:.4f} (Baseline : {ll_baseline_naive:.4f})")
    print(f"   ⚖️ Brier Score : {avg_brier:.4f}")
    
    # =========================================================================
    # MODÈLE 2 : RÉGRESSION (ÉCART DE BUTS EXACT)
    # =========================================================================
    print("\n[2/2] Entraînement Régression...")
    
    reg_params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 20  # <-- Déplacé ici
    }
    
    reg = xgb.XGBRegressor(**reg_params)
    reg.fit(
        X_train, y_reg_train,
        eval_set=[(X_train, y_reg_train), (X_test, y_reg_test)],
        verbose=False
    )
    
    mae = mean_absolute_error(y_reg_test, reg.predict(X_test))
    print(f"   🎯 Mean Absolute Error : {mae:.2f} buts par match")
    
    # Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/football_classifier.pkl')
    joblib.dump(reg, 'models/football_regressor.pkl')
    with open('models/football_features.txt', 'w') as f:
        f.write(','.join(features))
        
    print("\n💾 Modèles optimisés sauvegardés.")
    print("=" * 60)

if __name__ == "__main__":
    train_models()