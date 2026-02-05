import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, 
    log_loss, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features à utiliser pour l'entraînement
# On privilégie les DIFFÉRENCES (plus prédictives que les valeurs absolues)
FEATURES = [
    # Différences (les plus importantes!)
    'ELO_PRE_DIFF',      # Différence d'ELO
    'REST_DAYS_DIFF',    # Différence de repos
    'STREAK_DIFF',       # Différence de série
    'FORM_5_DIFF',       # Différence de forme récente
    'PTS_DIFF',          # Différence de points moyens
    'EFG_PCT_DIFF',      # Différence d'efficacité aux tirs
    'TOV_PCT_DIFF',      # Différence de turnovers (attention: inversé!)
    'NET_RTG_DIFF',      # Différence de net rating
    
    # Contexte individuel (peut aider)
    'HOME_IS_B2B',       # Domicile en B2B?
    'AWAY_IS_B2B',       # Extérieur en B2B?
    'HOME_REST_DAYS',    # Repos domicile
    'AWAY_REST_DAYS',    # Repos extérieur
]

# Features alternatives si certaines manquent
FEATURES_FALLBACK = [
    'ELO_PRE_DIFF',
    'REST_DAYS_DIFF',
    'HOME_AVG_PTS_10', 'AWAY_AVG_PTS_10',
    'HOME_AVG_EFG_PCT_10', 'AWAY_AVG_EFG_PCT_10',
    'HOME_ELO_PRE', 'AWAY_ELO_PRE',
    'HOME_REST_DAYS', 'AWAY_REST_DAYS',
]


# =============================================================================
# ÉVALUATION COMPLÈTE
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Modèle"):
    """
    Évaluation complète du modèle avec toutes les métriques importantes.
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    # Baseline (toujours prédire victoire domicile)
    baseline_acc = y_test.mean()  # % de victoires domicile
    
    # Affichage
    print(f"\n{'='*50}")
    print(f"📊 ÉVALUATION: {model_name}")
    print(f"{'='*50}")
    
    print(f"\n🎯 ACCURACY")
    print(f"   Modèle:     {accuracy:.2%}")
    print(f"   Baseline:   {baseline_acc:.2%} (toujours domicile)")
    print(f"   Gain:       {accuracy - baseline_acc:+.2%}")
    
    print(f"\n📈 DISCRIMINATION")
    print(f"   ROC-AUC:    {roc_auc:.3f}")
    print(f"   (0.5 = random, 0.6+ = bon, 0.7+ = très bon)")
    
    print(f"\n🎲 CALIBRATION (qualité des probabilités)")
    print(f"   Brier Score: {brier:.4f} (plus bas = mieux, <0.25 = OK)")
    print(f"   Log Loss:    {logloss:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 MATRICE DE CONFUSION")
    print(f"   Prédit →     Ext    Dom")
    print(f"   Réel Ext    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"   Réel Dom    {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Analyse par confiance
    print(f"\n🔍 ANALYSE PAR NIVEAU DE CONFIANCE")
    bins = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)]
    
    for low, high in bins:
        # Matchs où on prédit domicile avec cette confiance
        mask_home = (y_proba >= low) & (y_proba < high)
        # Matchs où on prédit extérieur avec cette confiance
        mask_away = ((1-y_proba) >= low) & ((1-y_proba) < high)
        
        mask = mask_home | mask_away
        if mask.sum() > 0:
            acc_bin = accuracy_score(y_test[mask], y_pred[mask])
            print(f"   Confiance {low:.0%}-{high:.0%}: {acc_bin:.1%} ({mask.sum()} matchs)")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier,
        'log_loss': logloss,
        'baseline': baseline_acc
    }


def print_feature_importance(model, feature_names, top_n=10):
    """Affiche l'importance des features."""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return
    
    # Trier par importance
    indices = np.argsort(importance)[::-1]
    
    print(f"\n🔑 TOP {top_n} FEATURES LES PLUS IMPORTANTES")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        bar = '█' * int(importance[idx] / importance[indices[0]] * 20)
        print(f"   {feature_names[idx]:25s} {importance[idx]:.3f} {bar}")


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train():
    """Pipeline d'entraînement principal."""
    
    print("=" * 60)
    print("🧠 NBA ORACLE - ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # --- 1. CHARGEMENT DES DONNÉES ---
    data_path = 'data/matchups.csv'
    
    if not os.path.exists(data_path):
        # Fallback sur l'ancien fichier si matchups n'existe pas
        print(f"⚠️  {data_path} non trouvé")
        print("   Lance d'abord: python src/process_data.py")
        return
    
    print(f"\n📂 Chargement de {data_path}...")
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    print(f"   {len(df)} matchups chargés")
    
    # --- 2. SÉLECTION DES FEATURES ---
    print(f"\n🔧 Sélection des features...")
    
    # Utiliser les features principales si disponibles
    available_features = [f for f in FEATURES if f in df.columns]
    
    if len(available_features) < 5:
        print(f"   ⚠️  Seulement {len(available_features)} features principales disponibles")
        print(f"   Utilisation des features alternatives...")
        available_features = [f for f in FEATURES_FALLBACK if f in df.columns]
    
    print(f"   Features utilisées ({len(available_features)}):")
    for f in available_features:
        print(f"      - {f}")
    
    # --- 3. PRÉPARATION DES DONNÉES ---
    # Supprimer les lignes avec NaN dans les features
    df_clean = df.dropna(subset=available_features + ['HOME_WIN'])
    print(f"\n   Matchups après nettoyage: {len(df_clean)}")
    
    X = df_clean[available_features]
    y = df_clean['HOME_WIN']
    
    # Split CHRONOLOGIQUE (pas de shuffle pour éviter le data leakage!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    
    print(f"   Train: {len(X_train)} matchs")
    print(f"   Test:  {len(X_test)} matchs")
    
    # --- 4. ENTRAÎNEMENT DE PLUSIEURS MODÈLES ---
    print("\n" + "-" * 40)
    print("🚀 Entraînement des modèles...")
    
    models = {
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
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\n   Entraînement {name}...")
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        
        # Feature importance pour ce modèle
        print_feature_importance(model, available_features)
        
        # Garder le meilleur
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = model
            best_name = name
    
    # --- 5. RÉSUMÉ COMPARATIF ---
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 60)
    print(f"\n{'Modèle':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Brier':>10}")
    print("-" * 55)
    
    for name, metrics in results.items():
        marker = " ⭐" if name == best_name else ""
        print(f"{name:<25} {metrics['accuracy']:>9.2%} {metrics['roc_auc']:>10.3f} {metrics['brier']:>10.4f}{marker}")
    
    print(f"\n🏆 Meilleur modèle: {best_name}")
    
    # --- 6. SAUVEGARDE ---
    print("\n💾 Sauvegarde du modèle...")
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/nba_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"   ✅ {model_path}")
    
    # Sauvegarder aussi les features utilisées
    features_path = 'models/features.txt'
    with open(features_path, 'w') as f:
        f.write('\n'.join(available_features))
    print(f"   ✅ {features_path}")
    
    # Sauvegarder les métriques
    metrics_path = 'models/metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Accuracy: {results[best_name]['accuracy']:.4f}\n")
        f.write(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}\n")
        f.write(f"Brier: {results[best_name]['brier']:.4f}\n")
        f.write(f"Baseline: {results[best_name]['baseline']:.4f}\n")
    print(f"   ✅ {metrics_path}")
    
    # --- 7. CONCLUSION ---
    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    
    gain = results[best_name]['accuracy'] - results[best_name]['baseline']
    print(f"\n   📈 Amélioration vs baseline: {gain:+.2%}")
    
    if results[best_name]['accuracy'] >= 0.60:
        print(f"   🎉 Excellent! Objectif de 60% atteint!")
    elif results[best_name]['accuracy'] >= 0.57:
        print(f"   👍 Bon modèle! Continue les améliorations.")
    else:
        print(f"   ⚠️  Modèle faible. Vérifie les données et features.")


if __name__ == "__main__":
    train()