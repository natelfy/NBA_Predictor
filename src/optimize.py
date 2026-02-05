import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer Optuna, sinon fallback sur GridSearch
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna non installé. Installation: pip install optuna")
    print("   Utilisation de GridSearch en fallback...")
    from sklearn.model_selection import GridSearchCV


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

RESULTS_FILE = 'models/optimization_results.json'


# =============================================================================
# OPTUNA OPTIMIZATION
# =============================================================================

class OptunaOptimizer:
    """Optimiseur utilisant Optuna (Bayesian optimization)."""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_models = {}
    
    def objective_logistic(self, trial):
        """Objectif pour Logistic Regression."""
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga',
            'max_iter': 2000,
            'random_state': 42
        }
        
        model = LogisticRegression(**params)
        model.fit(self.X_train, self.y_train)
        
        y_proba = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, y_proba)
    
    def objective_rf(self, trial):
        """Objectif pour Random Forest."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 40),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_proba = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, y_proba)
    
    def objective_gbm(self, trial):
        """Objectif pour Gradient Boosting."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 40),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        y_proba = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, y_proba)
    
    def optimize(self, model_type, n_trials=50):
        """Lance l'optimisation pour un type de modèle."""
        
        objectives = {
            'logistic': self.objective_logistic,
            'rf': self.objective_rf,
            'gbm': self.objective_gbm
        }
        
        if model_type not in objectives:
            raise ValueError(f"model_type inconnu: {model_type}")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objectives[model_type],
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        return study.best_params, study.best_value


# =============================================================================
# FALLBACK GRIDSEARCH
# =============================================================================

def gridsearch_optimize(X_train, y_train, model_type='logistic'):
    """Fallback si Optuna n'est pas installé."""
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=2000, solver='saga', random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 400, 600],
            'max_depth': [5, 7, 9],
            'min_samples_leaf': [10, 20, 30],
        }
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05, 0.1],
            'min_samples_leaf': [10, 20],
        }
    else:
        raise ValueError(f"model_type inconnu: {model_type}")
    
    grid = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    return grid.best_params_, grid.best_score_


# =============================================================================
# ÉVALUATION FINALE
# =============================================================================

def evaluate_optimized_model(model, X_test, y_test, model_name):
    """Évalue le modèle optimisé sur le set de test."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    baseline = y_test.mean()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier,
        'baseline': baseline,
        'gain_vs_baseline': accuracy - baseline
    }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def optimize(n_trials=50, quick=False):
    """
    Pipeline d'optimisation principal.
    
    Args:
        n_trials: Nombre d'essais par modèle (plus = meilleur mais plus long)
        quick: Si True, réduit le nombre d'essais
    """
    
    print("=" * 60)
    print("🔧 NBA ORACLE - OPTIMISATION DES HYPERPARAMÈTRES")
    print("=" * 60)
    
    if quick:
        n_trials = 20
        print(f"   Mode rapide: {n_trials} essais par modèle")
    else:
        print(f"   Mode complet: {n_trials} essais par modèle")
    
    # --- 1. CHARGEMENT ---
    data_path = 'data/matchups.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ {data_path} non trouvé")
        print("   Lance: python src/process_data.py")
        return
    
    print(f"\n📂 Chargement des données...")
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Features disponibles
    available = [f for f in FEATURES if f in df.columns]
    print(f"   {len(available)} features disponibles")
    
    # Nettoyage
    df_clean = df.dropna(subset=available + ['HOME_WIN'])
    print(f"   {len(df_clean)} matchups après nettoyage")
    
    X = df_clean[available]
    y = df_clean['HOME_WIN']
    
    # --- 2. SPLIT TEMPOREL ---
    # 70% train, 15% validation (pour optim), 15% test (évaluation finale)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"\n   Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # --- 3. OPTIMISATION ---
    print("\n" + "-" * 40)
    
    results = {}
    best_params_all = {}
    
    model_types = ['logistic', 'rf', 'gbm']
    model_names = {
        'logistic': 'Logistic Regression',
        'rf': 'Random Forest',
        'gbm': 'Gradient Boosting'
    }
    
    for model_type in model_types:
        print(f"\n🔍 Optimisation {model_names[model_type]}...")
        
        if OPTUNA_AVAILABLE:
            optimizer = OptunaOptimizer(X_train, y_train, X_val, y_val)
            best_params, best_score = optimizer.optimize(model_type, n_trials)
        else:
            # Combiner train et val pour GridSearch avec CV
            X_trainval = pd.concat([X_train, X_val])
            y_trainval = pd.concat([y_train, y_val])
            best_params, best_score = gridsearch_optimize(X_trainval, y_trainval, model_type)
        
        print(f"   ✅ Meilleur score validation: {best_score:.4f}")
        print(f"   📋 Params: {best_params}")
        
        best_params_all[model_type] = best_params
        
        # Entraîner le modèle final avec les meilleurs params
        if model_type == 'logistic':
            final_model = LogisticRegression(
                **best_params,
                solver='saga',
                max_iter=2000,
                random_state=42
            )
        elif model_type == 'rf':
            final_model = RandomForestClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gbm':
            final_model = GradientBoostingClassifier(
                **best_params,
                random_state=42
            )
        
        # Entraîner sur train + val, tester sur test
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        final_model.fit(X_trainval, y_trainval)
        
        # Évaluation finale
        metrics = evaluate_optimized_model(final_model, X_test, y_test, model_names[model_type])
        results[model_type] = {
            'params': best_params,
            'val_score': best_score,
            'metrics': metrics,
            'model': final_model
        }
    
    # --- 4. RÉSUMÉ ---
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'OPTIMISATION")
    print("=" * 60)
    
    print(f"\n{'Modèle':<25} {'Val AUC':>10} {'Test Acc':>10} {'Test AUC':>10}")
    print("-" * 55)
    
    best_model = None
    best_test_auc = 0
    best_model_type = None
    
    for model_type, data in results.items():
        metrics = data['metrics']
        val_auc = data['val_score']
        test_acc = metrics['accuracy']
        test_auc = metrics['roc_auc']
        
        marker = ""
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_model = data['model']
            best_model_type = model_type
            marker = " ⭐"
        
        print(f"{model_names[model_type]:<25} {val_auc:>10.4f} {test_acc:>9.2%} {test_auc:>10.4f}{marker}")
    
    # --- 5. MEILLEUR MODÈLE ---
    best_metrics = results[best_model_type]['metrics']
    best_params = results[best_model_type]['params']
    
    print(f"\n🏆 MEILLEUR MODÈLE: {model_names[best_model_type]}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.2%}")
    print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"   Brier:     {best_metrics['brier']:.4f}")
    print(f"   Baseline:  {best_metrics['baseline']:.2%}")
    print(f"   Gain:      {best_metrics['gain_vs_baseline']:+.2%}")
    
    # --- 6. SAUVEGARDE ---
    print("\n💾 Sauvegarde...")
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarder le meilleur modèle
    joblib.dump(best_model, 'models/nba_model.pkl')
    print("   ✅ models/nba_model.pkl")
    
    # Sauvegarder les features
    with open('models/features.txt', 'w') as f:
        f.write('\n'.join(available))
    print("   ✅ models/features.txt")
    
    # Sauvegarder les métriques
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Model: {model_names[best_model_type]} (Optimized)\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
        f.write(f"Brier: {best_metrics['brier']:.4f}\n")
        f.write(f"Baseline: {best_metrics['baseline']:.4f}\n")
        f.write(f"Optimized: True\n")
        f.write(f"Trials: {n_trials}\n")
    print("   ✅ models/metrics.txt")
    
    # Sauvegarder tous les résultats d'optimisation
    results_to_save = {}
    for model_type, data in results.items():
        results_to_save[model_type] = {
            'params': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                      for k, v in data['params'].items()},
            'val_score': float(data['val_score']),
            'test_accuracy': float(data['metrics']['accuracy']),
            'test_roc_auc': float(data['metrics']['roc_auc']),
        }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"   ✅ {RESULTS_FILE}")
    
    # --- 7. COMPARAISON AVANT/APRÈS ---
    print("\n" + "=" * 60)
    print("📈 COMPARAISON AVANT/APRÈS OPTIMISATION")
    print("=" * 60)
    
    # Charger les anciennes métriques si disponibles
    old_metrics_path = 'models/metrics_backup.txt'
    if os.path.exists('models/metrics.txt'):
        # Faire une copie de l'ancien
        import shutil
        if not os.path.exists(old_metrics_path):
            try:
                # Lire l'ancien accuracy
                with open('models/metrics.txt', 'r') as f:
                    old_content = f.read()
                    for line in old_content.split('\n'):
                        if line.startswith('Accuracy:'):
                            old_acc = float(line.split(':')[1].strip())
                            print(f"\n   Avant optimisation: {old_acc:.2%}")
                            print(f"   Après optimisation: {best_metrics['accuracy']:.2%}")
                            print(f"   Amélioration:       {best_metrics['accuracy'] - old_acc:+.2%}")
            except:
                pass
    
    print(f"\n✅ OPTIMISATION TERMINÉE!")
    print(f"   Ton modèle optimisé est prêt dans models/nba_model.pkl")
    print(f"   Lance: streamlit run app.py")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimisation NBA Oracle')
    parser.add_argument('--trials', type=int, default=50,
                       help='Nombre d\'essais par modèle (défaut: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Mode rapide (20 essais)')
    
    args = parser.parse_args()
    
    optimize(n_trials=args.trials, quick=args.quick)