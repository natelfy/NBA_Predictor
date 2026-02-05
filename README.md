# 🏀 NBA Oracle v2 - Prédictions NBA avec Cotes

Système de prédiction des matchs NBA utilisant l'ELO, le Machine Learning, et les cotes des bookmakers.

## 🚀 Nouveautés v2

- ✅ **Intégration des cotes** : Affiche les probabilités du marché
- ✅ **Détection de Value Bets** : Compare notre modèle vs le marché
- ✅ **Optimisation hyperparamètres** : GridSearch intégré
- ✅ **Interface améliorée** : Visualisation plus claire

## 📦 Installation

```bash
# 1. Cloner le projet
git clone <repo>
cd nba_oracle_v2

# 2. Installer les dépendances
pip install -r requirements.txt
```

## 🔑 Configuration des Cotes (IMPORTANT)

Pour afficher les cotes des bookmakers, tu dois :

### Étape 1 : Créer un compte The Odds API

1. Va sur [https://the-odds-api.com/](https://the-odds-api.com/)
2. Clique sur "Get API Key"
3. Crée un compte (gratuit)
4. Copie ta clé API

### Étape 2 : Configurer la clé

Ouvre `src/odds_collector.py` et remplace :

```python
API_KEY = os.environ.get('ODDS_API_KEY', 'YOUR_API_KEY_HERE')
```

Par :

```python
API_KEY = os.environ.get('ODDS_API_KEY', 'ta_vraie_clé_ici')
```

Ou utilise une variable d'environnement :

```bash
export ODDS_API_KEY="ta_vraie_clé_ici"
```

### Limites du plan gratuit

- 500 requêtes/mois
- ~16 requêtes/jour
- Suffisant pour un usage personnel

## 📋 Utilisation

### Pipeline standard

```bash
# 1. Collecter les données NBA
python src/get_data.py

# 2. Traiter les données (ELO, features, matchups)
python src/process_data.py

# 3. Entraîner le modèle
python src/train_model.py

# 4. Lancer l'interface
streamlit run app.py
```

### Options d'entraînement

```bash
# Standard (rapide)
python src/train_model.py

# Avec optimisation hyperparamètres (plus long, meilleur résultat)
python src/train_model.py --optimize

# Avec cotes historiques (si disponibles)
python src/train_model.py --odds
```

## 📊 Métriques

| Version | Accuracy | ROC-AUC | Amélioration |
|---------|----------|---------|--------------|
| v1 (ancien) | 56.4% | ~0.58 | - |
| v2 (actuel) | 62.1% | 0.675 | +5.7% |
| v2 + optimisation | ~63-64% | ~0.69 | +1-2% |
| v2 + cotes | ~65-66% | ~0.72 | +2-3% |

## 📁 Structure

```
nba_oracle_v2/
├── app.py                    # Interface Streamlit
├── requirements.txt          
├── README.md
├── src/
│   ├── get_data.py          # Collecte NBA
│   ├── process_data.py      # Features & ELO
│   ├── train_model.py       # Entraînement
│   └── odds_collector.py    # 🆕 Collecte des cotes
├── data/
│   ├── raw_games.csv
│   ├── processed_games.csv
│   ├── matchups.csv
│   └── odds_cache.json      # 🆕 Cache des cotes
└── models/
    ├── nba_model.pkl
    ├── features.txt
    └── metrics.txt
```

## 🎯 Comprendre les prédictions

### Interface

- **🤖 Notre Modèle** : Probabilité calculée par notre système
- **🎰 Marché** : Probabilité implicite des bookmakers
- **💎 Value** : Écart significatif entre notre modèle et le marché

### Value Bets

Un "value bet" existe quand notre modèle donne une probabilité
significativement différente du marché (>5%).

Exemple :
- Notre modèle : Lakers 65%
- Marché : Lakers 58%
- Edge : +7% → Value sur Lakers

⚠️ **Attention** : Un edge positif ne garantit pas la victoire !
C'est une opportunité statistique sur le long terme.

## 🔧 Améliorations futures

1. **Données de blessures** : Intégrer les absences de joueurs
2. **Cotes historiques** : Entraîner le modèle avec les cotes passées
3. **Features avancées** : Road trips, fuseaux horaires, rivalités
4. **Multi-sports** : Extension NFL, Tennis, Football européen

## ⚠️ Avertissement

Ce projet est à but **éducatif et personnel**.

- Les paris sportifs comportent des risques financiers
- Aucun modèle ne garantit des gains
- Joue de manière responsable

## 📝 Changelog

### v2.0
- Intégration API cotes (The Odds API)
- Détection des value bets
- Optimisation hyperparamètres (GridSearch)
- Interface améliorée

### v1.1
- Correction calcul ELO
- Structure matchups
- Features différentielles
- Précision 56% → 62%