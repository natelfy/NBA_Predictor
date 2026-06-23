import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

def train_models():
    print("=" * 60)
    print("🧠 ENTRAÎNEMENT PURIFIÉ (V3 - VALIDATION SÉPARÉE & CALIBRATION)")
    print("=" * 60)

    # 1. Chargement
    df = pd.read_csv('data/football_matchups.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')
    
    features = [c for c in df.columns if (c.endswith('_DIFF') and not c.startswith('TARGET_')) or c == 'T1_IS_HOST']
    X = df[features]
    
    target_map = {'2': 0, 'X': 1, '1': 2}
    y_class = df['TARGET_1X2'].astype(str).map(target_map)
    y_reg = df['TARGET_GOAL_DIFF']
    
    # 2. Split Strict : 70% Train / 15% Validation / 15% Test (La fin du Leakage)
    n = len(df)
    train_idx, val_idx = int(n * 0.70), int(n * 0.85)
    
    X_train, y_class_train, y_reg_train = X.iloc[:train_idx], y_class.iloc[:train_idx], y_reg.iloc[:train_idx]
    X_val, y_class_val, y_reg_val = X.iloc[train_idx:val_idx], y_class.iloc[train_idx:val_idx], y_reg.iloc[train_idx:val_idx]
    X_test, y_class_test, y_reg_test = X.iloc[val_idx:], y_class.iloc[val_idx:], y_reg.iloc[val_idx:]
    
    print(f"📊 Dataset : {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test")

    # =========================================================================
    # MODÈLE 1 : CLASSIFICATION & CALIBRATION ISOTONIQUE
    # =========================================================================
    print("\n[1/2] Classification avec Early Stopping pur...")
    clf = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        max_depth=4, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=20
    )
    
    # Le modèle s'arrête en regardant le set de Validation
    clf.fit(X_train, y_class_train, eval_set=[(X_val, y_class_val)], verbose=False)
    
    # CALIBRATION POST-HOC (Adaptation Scikit-Learn moderne)
    print("⚖️ Application de la Calibration Isotonique...")
    try:
        # Pour les versions récentes de scikit-learn (API 1.6+)
        from sklearn.frozen import FrozenEstimator
        calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(clf), method='isotonic')
    except ImportError:
        # Pour les anciennes versions
        calibrated_clf = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
        
    calibrated_clf.fit(X_val, y_class_val)
    
    class_probs = calibrated_clf.predict_proba(X_test)
    class_preds = np.argmax(class_probs, axis=1)
    
    ll = log_loss(y_class_test, class_probs)
    avg_brier = np.mean([brier_score_loss((y_class_test == k).astype(int), class_probs[:, k]) for k in range(3)])
    
    print(f"   🎯 Log Loss Honnête : {ll:.4f}")
    print(f"   ⚖️ Brier Score (Calibré) : {avg_brier:.4f}")

    # =========================================================================
    # MODÈLE 2 : RÉGRESSION
    # =========================================================================
    print("\n[2/2] Régression des Buts...")
    reg = xgb.XGBRegressor(
        objective='reg:squarederror', max_depth=3, learning_rate=0.05,
        n_estimators=500, subsample=0.8, random_state=42, early_stopping_rounds=20
    )
    reg.fit(X_train, y_reg_train, eval_set=[(X_val, y_reg_val)], verbose=False)
    print(f"   🎯 MAE : {mean_absolute_error(y_reg_test, reg.predict(X_test)):.2f} buts/match")

    # =========================================================================
    # SAUVEGARDE
    # =========================================================================
    os.makedirs('models', exist_ok=True)
    joblib.dump(calibrated_clf, 'models/football_classifier_calibrated.pkl')
    joblib.dump(reg, 'models/football_regressor.pkl')
    with open('models/football_features.txt', 'w') as f:
        f.write(','.join(features))
        
    print("\n💾 Modèles enregistrés. Le pipeline est prêt pour l'EV Score Exact.")

if __name__ == "__main__":
    train_models()