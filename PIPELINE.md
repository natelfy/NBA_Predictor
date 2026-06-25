# 🗓️ Pipeline quotidienne — Football Oracle (MPP World Cup 2026)

Ce que tu lances **chaque jour, dans cet ordre, AVANT de pronostiquer**.

## 0. Mettre à jour les résultats (manuel)
Ajoute les scores de la veille dans **`data/wc_2026_results.csv`**
(format : `date,home_team,away_team,home_score,away_score`).
C'est la seule étape manuelle, et c'est elle qui alimente tout le reste.

## 1. Recalculer l'état des équipes (LE plus important)
```bash
python src/process_football_data.py
```
→ régénère `data/processed_football_games.csv` en **injectant les résultats WC**.
C'est ce qui met à jour **l'ELO et la forme** de chaque équipe pour l'inférence.
**Si tu sautes cette étape, l'appli prédit avec une forme périmée.**

## 2. Recalculer les forces (attaque/défense ajustées adversaire)
```bash
python src/fit_team_ratings.py
```
→ régénère `data/team_ratings.json` (utilisé pour les **scores exacts**, lit `processed`).

## 3. Mettre à jour le radar + la calibration live
```bash
python src/tracker_football.py
```
→ régénère `data/mpp_live_tracking.csv` et **imprime le radar** (log loss, calibration
des nuls, biais favoris). Cette étape **alimente la calibration live** des probas (correction
du biais nuls) que l'appli applique automatiquement. Lis le verdict du radar au passage.

## 4. Prédire
```bash
streamlit run app.py
```
→ page **MPP World Cup** : saisis pour chaque match les **cotes** + le **% de la foule**,
puis **« Calculer la stratégie du jour »**. Coche **⚔️ Mode chasse** seulement en fin de
tournoi si tu es encore loin.

---

## ⚙️ À NE PAS lancer tous les jours (modèle figé pendant la WC)
```bash
python src/build_football_matchups.py   # reconstruit le dataset d'entraînement
python src/train_football_models.py     # ré-entraîne + calibre + écrit metrics_football.txt
```
`raw_international_games.csv` s'arrête avant la WC et **ne change pas** pendant le tournoi →
ré-entraîner ne change rien au quotidien. À ne relancer que si tu modifies les features ou
ajoutes des données pré-WC. *(Le modèle apprend sur ~49 000 matchs ; les ~50 matchs WC = 0,1 %
→ aucun impact mesurable à l'entraînement. Les résultats WC servent à l'INFÉRENCE via l'étape 1,
et à la CALIBRATION via l'étape 3 — pas à l'entraînement.)*

---

## TL;DR (copier-coller chaque jour, après avoir édité wc_2026_results.csv)
```bash
python src/process_football_data.py
python src/fit_team_ratings.py
python src/tracker_football.py
streamlit run app.py
```

## Rappel stratégie (la vérité)
Écart au 3e ~ -588 et en hausse, edge modèle mince face à ~599 joueurs → **top-1 hors de portée,
top-3 = long shot** atteignable seulement par une série de bons paris contrarian. Ton meilleur
levier mesuré : **les nuls que la foule fuit** (le modèle les sous-estime, surtout face aux gros
favoris). Joue 2-3 vrais coups contrarian (edge fiable, pas les ⚠️ « écart énorme ») par jour,
et relance ce cycle chaque journée.
