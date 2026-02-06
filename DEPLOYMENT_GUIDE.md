# 🚀 GUIDE DE DÉPLOIEMENT - NBA Oracle

## Vue d'ensemble

Ce guide te permet de déployer ton outil NBA Oracle en ligne avec :
- **Streamlit Cloud** : Hébergement gratuit de l'interface web
- **GitHub Actions** : Mise à jour automatique quotidienne des données

Temps estimé : **15-20 minutes**

---

## 📋 ÉTAPE 1 : Préparer ton projet

### 1.1 Structure requise

Assure-toi que ton projet a cette structure :

```
nba-oracle/
├── app.py                          ← Interface (fourni)
├── requirements.txt                ← Dépendances (fourni)
├── .gitignore                      ← Fichiers à ignorer (fourni)
├── README.md                       ← Documentation (fourni)
├── .streamlit/
│   └── config.toml                 ← Config Streamlit (fourni)
├── .github/
│   └── workflows/
│       └── daily-update.yml        ← Automatisation (fourni)
├── src/
│   ├── __init__.py
│   ├── get_data.py                 ← Tes fichiers existants
│   ├── process_data.py
│   ├── train_model.py
│   └── odds_collector.py
├── data/
│   ├── raw_games.csv               ← Tes données existantes
│   ├── processed_games.csv
│   └── matchups.csv
└── models/
    ├── nba_model.pkl               ← Ton modèle existant
    ├── features.txt
    └── metrics.txt
```

### 1.2 Fusionner les fichiers

1. Télécharge le ZIP `nba_oracle_auto.zip`
2. Extrais-le
3. Copie les fichiers dans ton projet existant :
   - `.github/` (dossier complet)
   - `.streamlit/` (dossier complet)
   - `app.py` (remplace l'ancien)
   - `.gitignore`
   - `README.md`

---

## 📤 ÉTAPE 2 : Créer le repo GitHub

### 2.1 Sur GitHub.com

1. Va sur [github.com/new](https://github.com/new)
2. Nom du repo : `nba-oracle` (ou ce que tu veux)
3. Description : "Prédictions NBA avec Machine Learning"
4. **Public** (requis pour Streamlit Cloud gratuit)
5. NE COCHE PAS "Add a README" (tu en as déjà un)
6. Clique "Create repository"

### 2.2 Dans ton terminal

```bash
# Va dans ton dossier projet
cd ~/Desktop/NBA_Predictor

# Initialise Git (si pas déjà fait)
git init

# Ajoute tous les fichiers
git add .

# Premier commit
git commit -m "🚀 Initial commit - NBA Oracle"

# Connecte à GitHub (remplace USERNAME par ton nom GitHub)
git remote add origin https://github.com/USERNAME/nba-oracle.git

# Push
git branch -M main
git push -u origin main
```

---

## ☁️ ÉTAPE 3 : Déployer sur Streamlit Cloud

### 3.1 Connexion

1. Va sur [share.streamlit.io](https://share.streamlit.io)
2. Clique "Sign in with GitHub"
3. Autorise Streamlit à accéder à tes repos

### 3.2 Déploiement

1. Clique "New app"
2. **Repository** : Sélectionne `USERNAME/nba-oracle`
3. **Branch** : `main`
4. **Main file path** : `app.py`
5. Clique "Deploy!"

### 3.3 Attendre

Le déploiement prend 2-5 minutes. Tu verras :
- Installation des dépendances
- Lancement de l'app

### 3.4 Ton URL

Une fois déployé, tu auras une URL comme :
```
https://username-nba-oracle-app-xxxxx.streamlit.app
```

C'est ton lien à partager ! 🎉

---

## ⚙️ ÉTAPE 4 : Configurer les secrets (optionnel)

Si tu veux les cotes des bookmakers :

### 4.1 Dans Streamlit Cloud

1. Va sur ton app dans le dashboard Streamlit
2. Clique "Settings" (⚙️)
3. Va dans "Secrets"
4. Ajoute :

```toml
ODDS_API_KEY = "ta_clé_the_odds_api"
```

5. Clique "Save"

---

## 🔄 ÉTAPE 5 : Vérifier l'automatisation

### 5.1 GitHub Actions

Les mises à jour automatiques sont configurées pour s'exécuter :
- **Tous les jours à 10h UTC** (11h Paris, 5h New York)

Pour vérifier :
1. Va sur ton repo GitHub
2. Clique sur l'onglet "Actions"
3. Tu verras le workflow "NBA Oracle - Mise à jour quotidienne"

### 5.2 Test manuel

Pour tester maintenant sans attendre demain :
1. Va dans "Actions"
2. Clique sur "NBA Oracle - Mise à jour quotidienne"
3. Clique "Run workflow" > "Run workflow"
4. Attends 2-3 minutes
5. Vérifie que ça passe au vert ✅

---

## 🔧 DÉPANNAGE

### "ModuleNotFoundError"

Vérifie que `requirements.txt` contient toutes les dépendances.

### "FileNotFoundError: models/nba_model.pkl"

Le modèle n'a pas été push. Vérifie :
```bash
git add models/
git commit -m "Add model files"
git push
```

### GitHub Actions échoue

1. Va dans Actions > le workflow qui a échoué
2. Clique dessus pour voir les logs
3. L'erreur est généralement dans l'étape rouge

### L'app ne se met pas à jour

Streamlit Cloud détecte automatiquement les changements Git.
Après un push, attends 1-2 minutes et rafraîchis la page.

---

## 📊 RÉSUMÉ

| Composant | Service | Coût |
|-----------|---------|------|
| Code | GitHub | Gratuit |
| Interface | Streamlit Cloud | Gratuit |
| Automatisation | GitHub Actions | Gratuit (2000 min/mois) |
| Cotes | The Odds API | Gratuit (500 req/mois) |

**Total : 0€/mois** 🎉

---

## 🎯 CHECKLIST FINALE

- [ ] Fichiers copiés dans le projet
- [ ] Repo GitHub créé
- [ ] Code pushé sur GitHub
- [ ] App déployée sur Streamlit Cloud
- [ ] URL de l'app notée
- [ ] Test manuel du workflow GitHub Actions
- [ ] (Optionnel) Clé API cotes configurée

---

## 🆘 BESOIN D'AIDE ?

Si tu bloques sur une étape, copie-colle le message d'erreur et je t'aide !
