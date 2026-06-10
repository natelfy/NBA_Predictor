"""
Football Oracle - Diagnostic & Calibration des probabilites 1X2
================================================================

POURQUOI CE SCRIPT
------------------
Le moteur strategique (mpp_oracle.py) calcule EV = prob * points_MPP.
Cette EV n'a de sens que si `prob` est CALIBREE : quand le modele annonce 60%,
l'evenement doit se produire ~60% du temps. Or XGBoost (multi:softprob) ne
garantit PAS cette propriete ; ses scores sont typiquement sur-confiants et
tasses. On le mesure, on corrige, on re-mesure.

DEMARCHE (mesurer avant d'agir)
  1. Reconstruit le modele de base avec EXACTEMENT les memes hyperparametres
     que train_football_models.py -> on mesure TON modele, pas un autre.
  2. Mesure la calibration AVANT : Brier one-vs-rest par classe, ECE, table
     de fiabilite. (L'accuracy est volontairement ignoree : trop instable.)
  3. Calibre via sigmoide (Platt) ET isotonic sur un bloc temporel DEDIE,
     que le modele de base n'a jamais vu (sinon fuite).
  4. Re-mesure APRES sur un bloc de test encore distinct, compare les 3
     variantes (brute / sigmoid / isotonic).
  5. Sauvegarde le meilleur classifieur calibre SANS ecraser le modele existant.

ISOLATION : aucune lecture/ecriture de fichier du projet NBA.
  Entree : data/football_matchups.csv (produit par process_football_data.py)
  Sortie : models/football_classifier_calibrated.pkl (+ rapport console)

Usage :
    python src/calibrate_football.py
    python src/calibrate_football.py --method isotonic   # forcer une methode
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("XGBoost requis : pip install xgboost")

DATA_PATH = "data/football_matchups.csv"
OUT_PATH = "models/football_classifier_calibrated.pkl"

# Memes parametres que train_football_models.py : on mesure le modele reel.
CLF_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

CLASS_LABELS = {0: "T2 gagne", 1: "Nul", 2: "T1 gagne"}


# ============================================================================
# METRIQUES DE CALIBRATION (orientees EV : on regarde chaque classe separement)
# ============================================================================

def ovr_brier_per_class(y_true, proba, classes):
    """Brier score one-vs-rest pour chaque classe. C'est la metrique EV-pertinente :
    elle mesure la qualite de CHAQUE proba (T1/Nul/T2), pas un agregat."""
    out = {}
    for i, c in enumerate(classes):
        y_bin = (np.asarray(y_true) == c).astype(int)
        out[c] = float(np.mean((proba[:, i] - y_bin) ** 2))
    return out


def ece_per_class(y_true, proba, classes, n_bins=10):
    """Expected Calibration Error one-vs-rest : ecart moyen |confiance - frequence|,
    pondere par le nombre de matchs dans chaque bin. 0 = parfaitement calibre."""
    out = {}
    edges = np.linspace(0, 1, n_bins + 1)
    for i, c in enumerate(classes):
        y_bin = (np.asarray(y_true) == c).astype(int)
        p = proba[:, i]
        ece = 0.0
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p <= hi) if b == n_bins - 1 else (p >= lo) & (p < hi)
            if mask.sum() > 0:
                ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y_bin[mask].mean())
        out[c] = float(ece)
    return out


def print_reliability(y_true, proba, classes, class_idx, n_bins=10):
    """Table de fiabilite texte pour une classe : proba predite vs frequence reelle."""
    c = classes[class_idx]
    y_bin = (np.asarray(y_true) == c).astype(int)
    p = proba[:, class_idx]
    edges = np.linspace(0, 1, n_bins + 1)
    print(f"\n   Fiabilite - classe '{CLASS_LABELS.get(c, c)}'")
    print(f"   {'bin proba':>14} {'predit':>8} {'reel':>8} {'n':>6}")
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi) if b == n_bins - 1 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        flag = "  <-- ecart" if abs(p[mask].mean() - y_bin[mask].mean()) > 0.10 else ""
        print(f"   {f'[{lo:.1f}-{hi:.1f}]':>14} {p[mask].mean():>8.1%} "
              f"{y_bin[mask].mean():>8.1%} {n:>6}{flag}")


def summarize(name, y_true, proba, classes):
    ll = log_loss(y_true, proba, labels=classes)
    brier = ovr_brier_per_class(y_true, proba, classes)
    ece = ece_per_class(y_true, proba, classes)
    mean_brier = float(np.mean(list(brier.values())))
    mean_ece = float(np.mean(list(ece.values())))
    print(f"\n[{name}]")
    print(f"   Log loss (global)    : {ll:.4f}")
    print(f"   Brier OvR moyen      : {mean_brier:.4f}")
    print(f"   ECE OvR moyen        : {mean_ece:.4f}")
    print(f"   Brier par classe     : " +
          " | ".join(f"{CLASS_LABELS.get(c, c)}={brier[c]:.4f}" for c in classes))
    return {"log_loss": ll, "mean_brier": mean_brier, "mean_ece": mean_ece}


# ============================================================================
# CALIBRATEUR (compatible sklearn < 1.6 et >= 1.6)
# ============================================================================

def make_calibrator(base_fitted, method):
    """CalibratedClassifierCV sur modele deja entraine. Gere la depreciation
    de cv='prefit' (remplace par FrozenEstimator en sklearn >= 1.6)."""
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        return CalibratedClassifierCV(FrozenEstimator(base_fitted), method=method)
    except Exception:
        return CalibratedClassifierCV(base_fitted, method=method, cv="prefit")


# ============================================================================
# PIPELINE
# ============================================================================

def run(forced_method=None):
    print("=" * 64)
    print("FOOTBALL ORACLE - DIAGNOSTIC & CALIBRATION DES PROBABILITES 1X2")
    print("=" * 64)

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} introuvable. Lance d'abord process_football_data.py")

    df = pd.read_csv(DATA_PATH)
    df["MATCH_DATE"] = pd.to_datetime(df["MATCH_DATE"])
    df = df.sort_values("MATCH_DATE").reset_index(drop=True)

    # Selection des features identique a train_football_models.py
    features = [c for c in df.columns
                if (c.endswith("_DIFF") and not c.startswith("TARGET_")) or c == "T1_IS_HOST"]
    X = df[features]
    y = df["TARGET_1X2"].astype(int)
    classes = np.array([0, 1, 2])

    # --- Split temporel STRICT en 3 blocs : train / calibration / test ---
    # Le calibrateur DOIT etre ajuste sur des donnees que le modele de base
    # n'a pas vues, et evalue sur un 3e bloc encore distinct.
    n = len(df)
    i_train = int(n * 0.70)
    i_calib = int(n * 0.85)

    X_tr, y_tr = X.iloc[:i_train], y.iloc[:i_train]
    X_cal, y_cal = X.iloc[i_train:i_calib], y.iloc[i_train:i_calib]
    X_te, y_te = X.iloc[i_calib:], y.iloc[i_calib:]

    print(f"\nEchantillon : {n} matchs | {len(features)} features")
    print(f"   Train      : {len(X_tr):>5}  ({df['MATCH_DATE'].iloc[0].date()} -> {df['MATCH_DATE'].iloc[i_train-1].date()})")
    print(f"   Calibration: {len(X_cal):>5}  ({df['MATCH_DATE'].iloc[i_train].date()} -> {df['MATCH_DATE'].iloc[i_calib-1].date()})")
    print(f"   Test       : {len(X_te):>5}  ({df['MATCH_DATE'].iloc[i_calib].date()} -> {df['MATCH_DATE'].iloc[-1].date()})")

    # [A VERIFIER] L'isotonic exige un echantillon de calibration suffisant.
    if len(X_cal) < 500:
        print(f"\n   [A VERIFIER] Bloc de calibration = {len(X_cal)} matchs (< 500). "
              f"L'isotonic risque le surapprentissage ; privilegie la sigmoide.")
    # [A VERIFIER] Distribution des classes : si tres desequilibree, la
    # calibration multiclasse one-vs-rest peut etre instable.
    print(f"   Repartition classes (test) : " +
          " | ".join(f"{CLASS_LABELS[c]}={int((y_te==c).sum())}" for c in classes))

    # --- Modele de base (TON modele) ---
    print("\n" + "-" * 64)
    print("Entrainement du modele de base (parametres train_football_models.py)...")
    base = xgb.XGBClassifier(**CLF_PARAMS)
    base.fit(X_tr, y_tr)

    proba_raw = base.predict_proba(X_te)
    res = {}
    res["brut (non calibre)"] = summarize("BRUT (non calibre)", y_te, proba_raw, classes)
    print_reliability(y_te, proba_raw, classes, class_idx=2)  # classe T1 gagne

    # --- Calibration ---
    methods = [forced_method] if forced_method else ["sigmoid", "isotonic"]
    calibrated_models = {}
    for method in methods:
        cal = make_calibrator(base, method)
        cal.fit(X_cal, y_cal)
        proba_cal = cal.predict_proba(X_te)
        # Reordonne les colonnes selon `classes` au cas ou classes_ differe
        order = [list(cal.classes_).index(c) for c in classes]
        proba_cal = proba_cal[:, order]
        res[method] = summarize(method.upper(), y_te, proba_cal, classes)
        calibrated_models[method] = cal

    # --- Verdict ---
    print("\n" + "=" * 64)
    print("COMPARAISON (critere principal : Brier OvR moyen, le plus bas gagne)")
    print("=" * 64)
    print(f"\n{'Variante':<22} {'LogLoss':>9} {'Brier':>9} {'ECE':>9}")
    print("-" * 52)
    for name, m in res.items():
        print(f"{name:<22} {m['log_loss']:>9.4f} {m['mean_brier']:>9.4f} {m['mean_ece']:>9.4f}")

    cal_methods = [m for m in res if m in calibrated_models]
    best = min(cal_methods, key=lambda k: res[k]["mean_brier"])
    gain = res["brut (non calibre)"]["mean_brier"] - res[best]["mean_brier"]

    print(f"\nMeilleure variante calibree : {best.upper()}")
    print(f"   Gain de Brier vs brut : {gain:+.4f}", end="  ")
    if gain > 0.0005:
        print("-> la calibration apporte qqch.")
    else:
        print("-> gain negligeable, le modele brut etait deja raisonnable.")

    # [A VERIFIER] On choisit la methode sur le set de TEST -> legerement
    # optimiste (selection sur la donnee d'evaluation). Le vrai juge de paix
    # sera la calibration mesuree sur les matchs reels du tournoi.
    print("\n   [A VERIFIER] La methode est choisie sur le test : estimation un peu")
    print("   optimiste. A confirmer sur les resultats reels du Mondial (tracking).")

    if gain > 0:
        os.makedirs("models", exist_ok=True)
        joblib.dump(calibrated_models[best], OUT_PATH)
        print(f"\nSauvegarde : {OUT_PATH}  (modele existant intact)")
        print("   Pour brancher dans mpp_oracle.py / mpp_bonus.py :")
        print(f"   clf = joblib.load('{OUT_PATH}')")
        print("   L'ordre des classes [0=T2, 1=Nul, 2=T1] est preserve -> indexation inchangee.")
    else:
        print("\nAucune sauvegarde : la calibration ne justifie pas de remplacer le modele.")

    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration Football Oracle")
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default=None,
                        help="Forcer une seule methode de calibration")
    args = parser.parse_args()
    run(forced_method=args.method)