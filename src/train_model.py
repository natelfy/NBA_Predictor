import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, 
    log_loss, confusion_matrix
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features SANS cotes (mode par défaut)
FEATURES_BASE = [
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

# Features AVEC cotes (si disponibles)
FEATURES_WITH_ODDS = FEATURES_BASE + [
    'IMPLIED_PROB_HOME',      # Probabilité implicite du marché
    'ODDS_ELO_DIFF',          # Différence entre notre ELO et les cotes
]

# Fallback si certaines features manquent
FEATURES_FALLBACK = [
    'ELO_PRE_DIFF',
    'REST_DAYS_DIFF',
    'HOME_AVG_PTS_10', 'AWAY_AVG_PTS_10',
    'HOME_ELO_PRE', 'AWAY_ELO_PRE',
    'HOME_REST_DAYS', 'AWAY_REST_DAYS',
]


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    """Évaluation complète du modèle."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    baseline_acc = y_test.mean()
    
    print(f"\n{'='*50}")
    print(f"📊 ÉVALUATION: {model_name}")
    print(f"{'='*50}")
    
    print(f"\n🎯 ACCURACY")
    print(f"   Modèle:     {accuracy:.2%}")
    print(f"   Baseline:   {baseline_acc:.2%}")
    print(f"   Gain:       {accuracy - baseline_acc:+.2%}")
    
    print(f"\n📈 DISCRIMINATION")
    print(f"   ROC-AUC:    {roc_auc:.3f}")
    
    print(f"\n🎲 CALIBRATION")
    print(f"   Brier:      {brier:.4f}")
    print(f"   Log Loss:   {logloss:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 CONFUSION")
    print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Analyse par confiance
    print(f"\n🔍 PAR CONFIANCE")
    for low, high in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)]:
        mask_home = (y_proba >= low) & (y_proba < high)
        mask_away = ((1-y_proba) >= low) & ((1-y_proba) < high)
        mask = mask_home | mask_away
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"   {low:.0%}-{high:.0%}: {acc:.1%} ({mask.sum()} matchs)")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier,
        'log_loss': logloss,
        'baseline': baseline_acc
    }


def print_feature_importance(model, feature_names, top_n=12):
    """Affiche l'importance des features."""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return
    
    indices = np.argsort(importance)[::-1]
    
    print(f"\n🔑 TOP FEATURES")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        bar = '█' * int(importance[idx] / importance[indices[0]] * 20)
        print(f"   {feature_names[idx]:25s} {importance[idx]:.3f} {bar}")


# =============================================================================
# OPTIMISATION HYPERPARAMÈTRES
# =============================================================================

def optimize_model(X_train, y_train, model_type='logistic'):
    """
    Optimise les hyperparamètres via GridSearch.
    
    Args:
        X_train, y_train: Données d'entraînement
        model_type: 'logistic', 'rf', ou 'gbm'
    
    Returns:
        Meilleur modèle trouvé
    """
    print(f"\n🔧 Optimisation {model_type}...")
    
    # Cross-validation temporelle (respecte l'ordre chronologique)
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['saga']
        }
    
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [6, 8, 10],
            'min_samples_leaf': [10, 20, 30]
        }
    
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'min_samples_leaf': [10, 20]
        }
    
    else:
        raise ValueError(f"model_type inconnu: {model_type}")
    
    grid = GridSearchCV(
        model, 
        param_grid, 
        cv=tscv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    print(f"   Meilleurs params: {grid.best_params_}")
    print(f"   Meilleur score CV: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


# =============================================================================
# ENTRAÎNEMENT PRINCIPAL
# =============================================================================

def train(optimize=False, use_odds=False):
    """
    Pipeline d'entraînement principal.
    
    Args:
        optimize: Si True, fait une recherche d'hyperparamètres
        use_odds: Si True, utilise les cotes comme features (si disponibles)
    """
    
    print("=" * 60)
    print("🧠 NBA ORACLE - ENTRAÎNEMENT DU MODÈLE")
    if optimize:
        print("   (Mode optimisation activé)")
    if use_odds:
        print("   (Mode cotes activé)")
    print("=" * 60)
    
    # --- 1. CHARGEMENT ---
    data_path = 'data/matchups.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ {data_path} non trouvé")
        print("   Lance: python src/process_data.py")
        return
    
    print(f"\n📂 Chargement...")
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    print(f"   {len(df)} matchups")
    
    # --- 2. SÉLECTION FEATURES ---
    print(f"\n🔧 Sélection features...")
    
    # Choisir le set de features
    if use_odds and 'IMPLIED_PROB_HOME' in df.columns:
        target_features = FEATURES_WITH_ODDS
        print("   Mode: AVEC cotes")
    else:
        target_features = FEATURES_BASE
        print("   Mode: SANS cotes")
    
    # Garder les features disponibles
    available = [f for f in target_features if f in df.columns]
    
    if len(available) < 5:
        print(f"   ⚠️ Fallback sur features alternatives")
        available = [f for f in FEATURES_FALLBACK if f in df.columns]
    
    print(f"   {len(available)} features:")
    for f in available:
        print(f"      - {f}")
    
    # --- 3. PRÉPARATION ---
    df_clean = df.dropna(subset=available + ['HOME_WIN'])
    print(f"\n   {len(df_clean)} matchups après nettoyage")
    
    X = df_clean[available]
    y = df_clean['HOME_WIN']
    
    # Split chronologique
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # --- 4. ENTRAÎNEMENT ---
    print("\n" + "-" * 40)
    
    if optimize:
        # Mode optimisation: GridSearch sur chaque type
        print("🚀 Optimisation des hyperparamètres...")
        
        models = {
            'Logistic (optimisé)': optimize_model(X_train, y_train, 'logistic'),
            'Random Forest (optimisé)': optimize_model(X_train, y_train, 'rf'),
            'Gradient Boosting (optimisé)': optimize_model(X_train, y_train, 'gbm'),
        }
    else:
        # Mode standard: paramètres par défaut améliorés
        print("🚀 Entraînement...")
        
        models = {
            'Logistic Regression': LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42
            ),
        }
        
        # Entraîner chaque modèle
        for name, model in models.items():
            print(f"\n   Training {name}...")
            model.fit(X_train, y_train)
    
    # --- 5. ÉVALUATION ---
    results = {}
    best_model = None
    best_auc = 0
    best_name = None
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        print_feature_importance(model, available)
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = model
            best_name = name
    
    # --- 6. RÉSUMÉ ---
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"\n{'Modèle':<30} {'Acc':>8} {'AUC':>8} {'Brier':>8}")
    print("-" * 55)
    
    for name, m in results.items():
        star = " ⭐" if name == best_name else ""
        print(f"{name:<30} {m['accuracy']:>7.2%} {m['roc_auc']:>8.3f} {m['brier']:>8.4f}{star}")
    
    print(f"\n🏆 Meilleur: {best_name}")
    print(f"   Accuracy: {results[best_name]['accuracy']:.2%}")
    print(f"   Gain vs baseline: {results[best_name]['accuracy'] - results[best_name]['baseline']:+.2%}")
    
    # --- 7. SAUVEGARDE ---
    print("\n💾 Sauvegarde...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(best_model, 'models/nba_model.pkl')
    print("   ✅ models/nba_model.pkl")
    
    with open('models/features.txt', 'w') as f:
        f.write('\n'.join(available))
    print("   ✅ models/features.txt")
    
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Accuracy: {results[best_name]['accuracy']:.4f}\n")
        f.write(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}\n")
        f.write(f"Brier: {results[best_name]['brier']:.4f}\n")
        f.write(f"Baseline: {results[best_name]['baseline']:.4f}\n")
        if optimize:
            f.write(f"Optimized: True\n")
    print("   ✅ models/metrics.txt")
    
    # --- 8. CONCLUSION ---
    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print("=" * 60)
    
    acc = results[best_name]['accuracy']
    if acc >= 0.65:
        print("   🏆 Excellent! Tu approches les pros!")
    elif acc >= 0.62:
        print("   🎉 Très bon! Objectif 60%+ atteint!")
    elif acc >= 0.58:
        print("   👍 Bon modèle, continue!")
    else:
        print("   ⚠️ Modèle faible, vérifie les données")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraînement NBA Oracle')
    parser.add_argument('--optimize', action='store_true', 
                       help='Active la recherche d\'hyperparamètres')
    parser.add_argument('--odds', action='store_true',
                       help='Utilise les cotes comme features')
    
    args = parser.parse_args()
    
    train(optimize=args.optimize, use_odds=args.odds)