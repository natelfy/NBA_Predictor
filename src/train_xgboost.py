import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Vérifier les imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost non installé: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM non installé: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES = [
    'ELO_PRE_DIFF',
    'REST_DAYS_DIFF',
    'STREAK_DIFF',
    'FORM_5_DIFF',
    'PTS_DIFF',
    'EFG_PCT_DIFF',
    'TOV_PCT_DIFF',
    'NET_RTG_DIFF',
    'HOME_IS_B2B',
    'AWAY_IS_B2B',
    'HOME_REST_DAYS',
    'AWAY_REST_DAYS',
]


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    """Évaluation complète."""
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)
    
    y_pred = (y_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    baseline = y_test.mean()
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name}")
    print(f"{'='*50}")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Baseline:  {baseline:.2%}")
    print(f"   Gain:      {accuracy - baseline:+.2%}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print(f"   Brier:     {brier:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier,
        'baseline': baseline
    }


def print_feature_importance(model, feature_names):
    """Affiche l'importance des features."""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return
    
    indices = np.argsort(importance)[::-1]
    
    print(f"\n🔑 TOP FEATURES")
    for i in range(min(8, len(feature_names))):
        idx = indices[i]
        bar = '█' * int(importance[idx] / importance[indices[0]] * 15)
        print(f"   {feature_names[idx]:22s} {importance[idx]:.3f} {bar}")


# =============================================================================
# OPTIMISATION OPTUNA
# =============================================================================

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimise XGBoost avec Optuna."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimise LightGBM avec Optuna."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value


# =============================================================================
# ENTRAÎNEMENT PRINCIPAL
# =============================================================================

def train(optimize=False, n_trials=50):
    """Pipeline d'entraînement XGBoost / LightGBM."""
    
    print("=" * 60)
    print("🚀 NBA ORACLE - XGBOOST / LIGHTGBM")
    print("=" * 60)
    
    if not XGBOOST_AVAILABLE and not LIGHTGBM_AVAILABLE:
        print("\n❌ Aucun modèle disponible!")
        print("   pip install xgboost lightgbm")
        return
    
    # --- 1. CHARGEMENT ---
    data_path = 'data/matchups.csv'
    
    if not os.path.exists(data_path):
        print(f"\n❌ {data_path} non trouvé")
        return
    
    print(f"\n📂 Chargement...")
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    available = [f for f in FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available + ['HOME_WIN'])
    
    X = df_clean[available]
    y = df_clean['HOME_WIN']
    print(f"   {len(X)} matchups, {len(available)} features")
    
    # --- 2. SPLIT TEMPOREL ---
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # --- 3. MODÈLES ---
    results = {}
    models = {}
    
    # Baseline: Logistic Regression (ton meilleur actuel)
    print("\n" + "-" * 40)
    print("📌 Logistic Regression (baseline actuel)")
    
    lr_model = LogisticRegression(C=0.001, penalty='l2', solver='saga', max_iter=2000, random_state=42)
    lr_model.fit(X_trainval, y_trainval)
    results['Logistic'] = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    models['logistic'] = lr_model
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "-" * 40)
        print("🔶 XGBoost")
        
        if optimize and OPTUNA_AVAILABLE:
            print(f"   Optimisation ({n_trials} essais)...")
            best_params, val_score = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials)
            print(f"   ✅ Meilleur AUC val: {val_score:.4f}")
            xgb_params = {**best_params, 'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}
        else:
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': 42,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_trainval, y_trainval)
        results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        print_feature_importance(xgb_model, available)
        models['xgboost'] = xgb_model
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n" + "-" * 40)
        print("🟢 LightGBM")
        
        if optimize and OPTUNA_AVAILABLE:
            print(f"   Optimisation ({n_trials} essais)...")
            best_params, val_score = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials)
            print(f"   ✅ Meilleur AUC val: {val_score:.4f}")
            lgb_params = {**best_params, 'random_state': 42, 'verbosity': -1}
        else:
            lgb_params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': -1
            }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_trainval, y_trainval)
        results['LightGBM'] = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        print_feature_importance(lgb_model, available)
        models['lightgbm'] = lgb_model
    
    # --- 4. COMPARAISON ---
    print("\n" + "=" * 60)
    print("📊 COMPARAISON FINALE")
    print("=" * 60)
    
    print(f"\n{'Modèle':<15} {'Accuracy':>10} {'ROC-AUC':>10} {'Brier':>10}")
    print("-" * 50)
    
    best_name = None
    best_auc = 0
    
    for name, m in results.items():
        marker = ""
        if m['roc_auc'] > best_auc:
            best_auc = m['roc_auc']
            best_name = name
            marker = " ⭐"
        print(f"{name:<15} {m['accuracy']:>9.2%} {m['roc_auc']:>10.4f} {m['brier']:>10.4f}{marker}")
    
    # --- 5. VERDICT ---
    print("\n" + "-" * 40)
    print(f"🏆 MEILLEUR: {best_name}")
    
    best_metrics = results[best_name]
    baseline_acc = results['Logistic']['accuracy']
    improvement = best_metrics['accuracy'] - baseline_acc
    
    if improvement > 0.001:
        print(f"   📈 Amélioration vs Logistic: {improvement:+.2%}")
    elif improvement < -0.001:
        print(f"   📉 Moins bon que Logistic: {improvement:+.2%}")
    else:
        print(f"   ➡️  Équivalent à Logistic")
    
    # --- 6. SAUVEGARDE ---
    if best_name != 'Logistic' and improvement > 0:
        print("\n💾 Sauvegarde du nouveau modèle...")
        os.makedirs('models', exist_ok=True)
        
        best_model = models[best_name.lower()]
        joblib.dump(best_model, 'models/nba_model.pkl')
        
        with open('models/features.txt', 'w') as f:
            f.write('\n'.join(available))
        
        with open('models/metrics.txt', 'w') as f:
            f.write(f"Model: {best_name}\n")
            f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
            f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
            f.write(f"Brier: {best_metrics['brier']:.4f}\n")
            f.write(f"Baseline: {best_metrics['baseline']:.4f}\n")
        
        print("   ✅ Modèle sauvegardé!")
    else:
        print("\n📌 Le modèle Logistic reste le meilleur.")
        print("   Pas de sauvegarde (ancien modèle conservé)")
    
    # --- 7. CONCLUSION ---
    print("\n" + "=" * 60)
    
    if improvement <= 0:
        print("""
💡 ANALYSE:

Le fait que Logistic Regression reste équivalent ou meilleur
signifie que tes features sont bien construites et que la 
relation avec la target est relativement linéaire.

XGBoost/LightGBM excellent quand il y a des interactions
non-linéaires complexes, ce qui n'est pas ton cas.

👍 C'est une BONNE nouvelle: ton feature engineering est solide!

Pour améliorer significativement, il faudrait:
• Des données supplémentaires (cotes historiques, blessures)
• Plus de features non-corrélées entre elles
        """)
    else:
        print(f"\n🎉 Nouveau record: {best_metrics['accuracy']:.2%}!")
    
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='Optimiser avec Optuna')
    parser.add_argument('--trials', type=int, default=50, help='Essais Optuna')
    args = parser.parse_args()
    
    train(optimize=args.optimize, n_trials=args.trials)