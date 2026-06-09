import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, brier_score_loss
import joblib
import os

def train_models():
    print("=" * 60)
    print("🧠 ENTRAÎNEMENT DES MODÈLES FOOTBALL ORACLE")
    print("=" * 60)

    # 1. Chargement des données
    data_path = 'data/football_matchups.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ {data_path} introuvable. Lance le script précédent.")
    
    df = pd.read_csv(data_path)
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    
    # 2. Définition des Features et Targets
    # On isole uniquement les différentiels et l'avantage terrain
    features = [c for c in df.columns if (c.endswith('_DIFF') and not c.startswith('TARGET_')) or c == 'T1_IS_HOST']
    
    X = df[features]
    y_class = df['TARGET_1X2']       # 0=T2 Win, 1=Draw, 2=T1 Win
    y_reg = df['TARGET_GOAL_DIFF']   # Ecart exact de buts
    
    # 3. Split Temporel (Éviter le leakage du futur vers le passé)
    # Entraînement sur les anciens matchs, Test sur les 20% les plus récents
    df = df.sort_values('MATCH_DATE')
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    
    print(f"📊 Dataset: {len(X_train)} matchs (Train) | {len(X_test)} matchs (Test)")
    print(f"⚙️ Features utilisées ({len(features)}): {', '.join(features)}")
    
    # =========================================================================
    # MODÈLE 1 : CLASSIFICATION (PROBABILITÉS 1X2)
    # =========================================================================
    print("\n[1/2] Entraînement du Modèle de Classification (Probabilités)...")
    
    # Paramètres conservateurs pour éviter l'overfitting sur le foot international
    clf_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_train, y_class_train)
    
    # Évaluation Classification
    class_preds = clf.predict(X_test)
    class_probs = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_class_test, class_preds)
    ll = log_loss(y_class_test, class_probs)
    
    print(f"   ✅ Accuracy : {acc:.1%} (Ne pas s'y fier, le foot a trop de variance)")
    print(f"   🎯 Log Loss : {ll:.4f} (Métrique critique pour l'EV ! Plus c'est proche de 0.9, mieux c'est)")
    
    # Feature Importance
    importance_clf = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 3 Features (Classification) :")
    for _, row in importance_clf.head(3).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.1%}")

    # =========================================================================
    # MODÈLE 2 : RÉGRESSION (ÉCART DE BUTS EXACT)
    # =========================================================================
    print("\n[2/2] Entraînement du Modèle de Régression (Écart de buts)...")
    
    reg_params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'random_state': 42
    }
    
    reg = xgb.XGBRegressor(**reg_params)
    reg.fit(X_train, y_reg_train)
    
    # Évaluation Régression
    reg_preds = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    
    print(f"   🎯 Mean Absolute Error : {mae:.2f} buts par match")
    
    # =========================================================================
    # SAUVEGARDE DES MODÈLES
    # =========================================================================
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/football_classifier.pkl')
    joblib.dump(reg, 'models/football_regressor.pkl')
    
    # Sauvegarder la liste des features pour le script de prédiction
    with open('models/football_features.txt', 'w') as f:
        f.write(','.join(features))
        
    print("\n💾 Modèles sauvegardés dans le dossier 'models/'")
    print("=" * 60)

if __name__ == "__main__":
    train_models()