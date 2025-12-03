import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    print("🧠 Entraînement du modèle (Stable Random Forest)...")
    
    # 1. Chargement SÉCURISÉ des données
    data_path = 'data/processed_games.csv'
    if not os.path.exists(data_path):
        # Essai à la racine si pas trouvé dans data/
        if os.path.exists('processed_games.csv'):
            data_path = 'processed_games.csv'
        else:
            print("❌ ERREUR : Impossible de trouver processed_games.csv")
            return

    df = pd.read_csv(data_path)

    # 2. FEATURES (Variables)
    # On s'assure d'utiliser les mêmes colonnes que celles créées par process_data.py
    features = [
        'IS_HOME', 
        'REST_DAYS',
        'ELO_PRE',           
        'AVG_EFG_PCT_10',
        'AVG_TOV_PCT_10',
        'AVG_FT_RATE_10',
        'AVG_OREB_PCT_10',
        'AVG_PTS_10'
    ]
    target = 'WIN'

    # Nettoyage ultime avant entraînement
    df = df.dropna(subset=features)

    X = df[features]
    y = df[target]

    # 3. SPLIT (On garde l'ordre chronologique pour ne pas tricher)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # 4. MODÈLE (Retour au Random Forest, plus robuste que le Boosting sur peu de données)
    # n_estimators=500 : On met beaucoup d'arbres pour lisser les résultats
    # max_depth=10 : On limite la profondeur pour éviter qu'il apprenne par coeur (Overfitting)
    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)

    # 5. ÉVALUATION
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Précision du modèle : {acc:.2%}")
    print("   (Note : >55% est bon pour du basket. >60% est excellent)")

    # 6. SAUVEGARDE SÉCURISÉE
    # On force la création du dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    save_path = 'models/nba_model.pkl'
    joblib.dump(model, save_path)
    print(f"✅ Modèle sauvegardé dans : {os.path.abspath(save_path)}")

if __name__ == "__main__":
    train()