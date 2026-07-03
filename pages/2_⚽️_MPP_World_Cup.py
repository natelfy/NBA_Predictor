import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath('src'))
from mpp_oracle import prepare_live_inference_features
import score_model as sm
import mpp_sim
import context_signals
import live_calibration
import jackpot

st.set_page_config(page_title="MPP World Cup Oracle", page_icon="⚽", layout="wide")
st.title("⚽ MPP Oracle — Stratégie du jour (P top-3)")
results_box = st.container()   # les résultats s'affichent ICI, en haut (visibles sans scroller)

OUTS = ['T1', 'NUL', 'T2']


# =====================================================================
# CHARGEMENT ENVIRONNEMENT (modèle calibré + ratings Poisson optionnels)
# =====================================================================
@st.cache_resource
def load_oracle_environment():
    clf = joblib.load('models/football_classifier_calibrated.pkl')
    reg = joblib.load('models/football_regressor.pkl')
    with open('models/football_features.txt', 'r') as f:
        features = [line.strip() for line in f.read().split(',') if line.strip()]
    df = pd.read_csv('data/processed_football_games.csv')
    df['MATCH_DATE'] = pd.to_datetime(df['MATCH_DATE'])
    ratings = sm.load_ratings('data/team_ratings.json')   # None si fit_team_ratings.py pas lancé
    return clf, reg, features, df, ratings


@st.cache_data
def load_upcoming_matches():
    try:
        sched = pd.read_csv('data/wc_2026_results.csv', names=['DATE', 'HOME', 'AWAY', 'HG', 'AG'], header=0)
        sched['DATE'] = pd.to_datetime(sched['DATE'])
        return sched[sched['HG'].isna() | sched['AG'].isna()].sort_values('DATE')
    except Exception as e:
        st.error(f"Erreur calendrier : {e}")
        return pd.DataFrame()


@st.cache_data
def load_context_signals():
    return context_signals.load_signals()   # (squad_power, fatigue_index) — contexte affiché


@st.cache_data
def load_calib():
    return live_calibration.load_calibration()   # corrige les biais mesurés sur les matchs WC joués


clf, reg, features_list, history_df, ratings = load_oracle_environment()
upcoming_matches = load_upcoming_matches()
squad_d, fatigue_d = load_context_signals()
CALIB = load_calib()
if clf is None:
    st.stop()


# =====================================================================
# SIDEBAR : une seule entrée essentielle
# =====================================================================
st.sidebar.header("🎯 Ta situation")
gap_3e = st.sidebar.number_input("Points à combler sur le 3e", value=505, step=10)
n_remaining = st.sidebar.number_input("Matchs restants (estim. jusqu'à la fin)",
                                      value=int(max(len(upcoming_matches), 30)), step=5, min_value=1)
chase = st.sidebar.checkbox("⚔️ Mode chasse (fin de tournoi)", value=False,
                            help="Maximise le PLAFOND de gain (variance) au lieu de l'objectif du jour. "
                                 "À n'utiliser que si tu es loin ET qu'il reste peu de matchs : ça augmente "
                                 "la (petite) proba de top-3 au prix d'une place moyenne plus basse.")
st.sidebar.caption(f"ratings : {'✅' if ratings else '❌'}")
st.sidebar.caption(live_calibration.describe(CALIB))

# Réglages internes (volontairement non exposés pour garder l'UI simple)
COMPETITOR = 'favori'   # la 'ligne du 3e' suit la foule (hypothèse réaliste)
TRIALS = 8000           # nb de simulations Monte-Carlo (compromis vitesse/précision)

if upcoming_matches.empty:
    st.info("🗓️ Aucun match à venir détecté.")
    st.stop()

next_date = upcoming_matches['DATE'].iloc[0]
daily = upcoming_matches[upcoming_matches['DATE'] == next_date]
st.subheader(f"🗓️ Grille du {next_date.strftime('%d/%m/%Y')}")
st.markdown("Pour chaque match : saisis les **cotes MPP** et le **% de la foule** (la répartition des "
            "prons que tu vois dans l'app). Puis clique sur **Calculer la stratégie du jour**.")


# =====================================================================
# HELPERS
# =====================================================================
def implied_from_cotes(cotes):
    inv = {k: 1.0 / max(cotes[k], 1e-9) for k in OUTS}
    s = sum(inv.values())
    return {k: inv[k] / s for k in OUTS}


def model_probs_and_lambdas(t1, t2):
    sub1 = history_df[history_df['TEAM_ID'] == t1].sort_values('MATCH_DATE')
    sub2 = history_df[history_df['TEAM_ID'] == t2].sort_values('MATCH_DATE')
    for t, sub in [(t1, sub1), (t2, sub2)]:
        if sub.empty:                       # guard noms (E3) : plus de fallback ELO=1500 silencieux
            st.warning(f"⚠️ '{t}' introuvable dans l'historique (problème de nom ?) — "
                       f"ELO/forme par défaut, proba peu fiable pour ce match.")
    elo_dict = {
        t1: (sub1.iloc[-1]['ELO_POST'] if not sub1.empty else 1500),
        t2: (sub2.iloc[-1]['ELO_POST'] if not sub2.empty else 1500),
    }
    X = prepare_live_inference_features(t1, t2, str(next_date.date()), history_df, elo_dict)
    X['T1_IS_HOST'] = 0
    X = X[features_list]
    probs = clf.predict_proba(X)[0]                      # [T2, NUL, T1]
    goal_diff = float(reg.predict(X)[0])
    p_model = {'T1': float(probs[2]), 'NUL': float(probs[1]), 'T2': float(probs[0])}
    p_model = live_calibration.correct(p_model, CALIB)   # corrige le biais nuls/favoris mesuré live
    if ratings is not None:
        lambdas = sm.lambdas_from_ratings(t1, t2, ratings, 0)
    else:
        lambdas = sm.lambdas_from_goal_diff(goal_diff)
    return p_model, lambdas


# =====================================================================
# SAISIE PAR MATCH (les valeurs persistent via les clés des widgets)
# =====================================================================
slate_inputs = []
for idx, match in daily.iterrows():
    t1, t2 = match['HOME'], match['AWAY']
    with st.expander(f"🔮 {t1} vs {t2}", expanded=True):
        c1, c2, c3 = st.columns(3)
        cote = {
            'T1': c1.number_input(f"Cote {t1}", value=50, step=1, key=f"cote_t1_{idx}"),
            'NUL': c2.number_input("Cote NUL", value=100, step=1, key=f"cote_nul_{idx}"),
            'T2': c3.number_input(f"Cote {t2}", value=50, step=1, key=f"cote_t2_{idx}"),
        }
        dflt = implied_from_cotes(cote)
        d1, d2, d3 = st.columns(3)
        crowd_pct = {
            'T1': d1.number_input(f"% foule {t1}", 0, 100, int(round(dflt['T1'] * 100)), key=f"cr_t1_{idx}"),
            'NUL': d2.number_input("% foule NUL", 0, 100, int(round(dflt['NUL'] * 100)), key=f"cr_nul_{idx}"),
            'T2': d3.number_input(f"% foule {t2}", 0, 100, int(round(dflt['T2'] * 100)), key=f"cr_t2_{idx}"),
        }
        slate_inputs.append({'t1': t1, 't2': t2, 'cote': cote, 'crowd_pct': crowd_pct})


# =====================================================================
# STRATÉGIE DU JOUR : modèle -> sim P(top-3) -> picks recommandés
# =====================================================================
if st.button("🎯 Calculer la stratégie du jour", type="primary"):
    with st.spinner("Calcul de la stratégie (modèle + simulations)…"):
        try:
            slate, rows = [], []
            for it in slate_inputs:
                t1, t2, cote = it['t1'], it['t2'], it['cote']
                p_model, lambdas = model_probs_and_lambdas(t1, t2)

                cp = it['crowd_pct']
                tot = max(cp['T1'] + cp['NUL'] + cp['T2'], 1)
                p_crowd = {k: cp[k] / tot for k in OUTS}     # normalisé à 1

                slate.append(mpp_sim.build_match(            # bonus via barème MPP (score_model.bonus_map)
                    name=f"{t1} vs {t2}", p_model=p_model, p_crowd=p_crowd,
                    cotes=cote, lambdas=lambdas))

                edge = {k: p_model[k] - p_crowd[k] for k in OUTS}
                mc = context_signals.match_context(t1, t2, squad_d, fatigue_d)
                sp1, sp2 = mc['t1']['squad_power'], mc['t2']['squad_power']
                talent = f"{sp1:.0f} vs {sp2:.0f}" if (sp1 is not None and sp2 is not None) else "n/a"
                rows.append({
                    'Match': f"{t1} vs {t2}",
                    'Modèle': f"T1 {p_model['T1']:.0%} / N {p_model['NUL']:.0%} / T2 {p_model['T2']:.0%}",
                    'Foule': f"T1 {p_crowd['T1']:.0%} / N {p_crowd['NUL']:.0%} / T2 {p_crowd['T2']:.0%}",
                    'Talent': talent,
                    'Edge NUL': f"{edge['NUL'] * 100:+.0f}",
                    'Contexte': mc['note'],
                })

            # Objectif PRORATISÉ : la part de l'écart au 3e à grignoter SUR LA JOURNÉE
            # (combler 505 d'un coup sur un seul jour est impossible -> P≈0 et choix non fiable).
            n_today = len(slate)
            daily_target = max(1.0, gap_3e * n_today / max(n_remaining, n_today))
            rec = mpp_sim.recommend(slate, gap=daily_target, trials=TRIALS, competitor=COMPETITOR,
                                    objective='ceiling' if chase else 'target')

            with results_box:
                p = rec['p_top3']
                st.markdown("## 🧮 Stratégie recommandée")
                cA, cB = st.columns([1, 2])
                cA.metric(f"P(objectif du jour ≥ +{daily_target:.0f})", f"{p:.1%}")
                cB.info(f"🎯 Objectif du jour ≈ **+{daily_target:.0f} pts nets** vs la foule "
                        f"(part de l'écart {gap_3e} réparti sur ~{int(n_remaining)} matchs restants). "
                        f"Meilleure stratégie : **{rec['best']}**. "
                        + ("⚠️ Reste un long shot — tiens la ligne et relance chaque jour."
                           if p < 0.15 else "Profil jouable : tiens cette ligne et relance chaque jour."))
                st.caption("📌 Calibration live : le biais sur les nuls **dépend du régime** — nuls sous-estimés "
                           "en poules (90'), mais **plus rares en phase finale (120')** où la prolongation casse "
                           "les égalités. L'outil se recalibre seul (pondéré par récence) : fie-toi à l'**edge "
                           "affiché**, pas à une règle figée.")

                st.markdown("### ✅ Tes picks du jour")
                for d in rec['picks']:
                    name = {'T1': d['match'].split(' vs ')[0], 'T2': d['match'].split(' vs ')[1],
                            'NUL': 'MATCH NUL'}[d['pick']]
                    sc = f"{d['score'][0]}-{d['score'][1]}"
                    if d['contrarian']:
                        suspect = d['edge'] > mpp_sim.EDGE_CAP   # cohérent avec l'exclusion du simulateur
                        tag = ("⚠️ écart ÉNORME vs foule → probable limite du modèle sur cette équipe (À VÉRIFIER), "
                               "pas forcément une vraie valeur" if suspect else "COUP CONTRARIAN")
                        line = (f"🎯 **{d['match']}** → **{name}** (score {sc}) · "
                                f"modèle {d['p_model']:.0%} vs foule {d['p_crowd']:.0%} "
                                f"(**edge {d['edge'] * 100:+.0f}**, cote {d['cote']}) — {tag}")
                        (st.warning if suspect else st.success)(line)
                    else:
                        st.write(f"• **{d['match']}** → {name} (score {sc}) · pas d'edge net → prudence "
                                 f"(modèle {d['p_model']:.0%} vs foule {d['p_crowd']:.0%}, cote {d['cote']})")

                st.markdown("### Comparaison des stratégies")
                comp_rows = [{'Stratégie': n, 'P(top-3)': f"{r['p_top3']:.1%}",
                              'Net médian': f"{r['net_median']:+.0f}", 'Net p90': f"{r['net_p90']:+.0f}",
                              'Anomalies/win': f"{r['avg_contrarian_hits_when_win']:.1f}"}
                             for n, r in rec['table'].items()]
                st.table(pd.DataFrame(comp_rows))

                with st.expander("Détail modèle vs foule par match"):
                    st.table(pd.DataFrame(rows))
                st.caption("Stratégie **LIGNE C** (hybride discipliné, régime 120') : on SUIT la foule par défaut, "
                           "et on ne dégaine un coup contrarian (nul OU vainqueur) que sur un **empilement de la "
                           "foule ≥ 75 %** ou un **edge ≥ +10**, à raison de **2 coups max par journée** — garde-fous "
                           "anti-erreur-modèle inchangés. x2 déjà dépensé. Relance à chaque journée.")
        except Exception as e:
            with results_box:
                st.error("❌ Le calcul a échoué — détail ci-dessous (copie-le moi pour que je corrige).")
                st.exception(e)


# =====================================================================
# 🎰 MODE JACKPOT : scores exacts (baroud d'honneur pour très gros écart)
# =====================================================================
if st.button("🎰 Mode JACKPOT : scores exacts à jouer (baroud d'honneur)"):
    with st.spinner("Recherche des scores exacts à plus fort upside…"):
        try:
            jslate = []
            for it in slate_inputs:
                t1, t2, cote = it['t1'], it['t2'], it['cote']
                _, lambdas = model_probs_and_lambdas(t1, t2)
                jslate.append(jackpot.build_match(
                    name=f"{t1} vs {t2}", cotes=cote, lambdas=lambdas))
            n_today = len(jslate)
            daily_target = max(1.0, gap_3e * n_today / max(n_remaining, n_today))
            jrec = jackpot.recommend(jslate, gap=daily_target, trials=TRIALS)

            with results_box:
                st.markdown("## 🎰 Mode JACKPOT — scores exacts")
                st.metric("P(objectif du jour) — indicatif, OPTIMISTE", f"{jrec['p_top3']:.1%}")
                st.caption(f"Profil retenu : **{jrec['best']}**. Le **bonus suit le barème officiel MPP** "
                           f"(paliers +20/30/50/70/**100 max**, selon la part des bons-résultat ayant le score) "
                           f"— estimé via P(score|issue), **rien à saisir**. ⚠️ Ce P est **optimiste** "
                           f"(simulateur à 1 concurrent, pas les ~598 autres) — le vrai chiffre est plus bas.")

                st.markdown("### 🎯 Scores exacts à jouer")
                for d in jrec['picks']:
                    i, j = d['score']
                    st.success(f"**{d['match']}** → **{i} - {j}** ({d['issue']}) · P {d['p']:.0%} · "
                               f"cote {d['cote']:.0f} + bonus {d['bonus']:.0f} = **{d['cote'] + d['bonus']:.0f} pts** si exact")
                    alts = " · ".join(f"{r['score'][0]}-{r['score'][1]} (P{r['p']*100:.0f}%·{r['cote']+r['bonus']:.0f}pts)"
                                      for r in d['alts'])
                    st.caption(f"alternatives : {alts}")

                st.markdown("### Comparaison des profils")
                st.table(pd.DataFrame([{'Profil': n, 'P(top-3) indic.': f"{r['p_top3']:.1%}",
                                        'Net p90': f"{r['net_p90']:+.0f}", 'Net max': f"{r['net_max']:+.0f}"}
                                       for n, r in jrec['table'].items()]))
                st.caption("💡 Clé : le bonus est **plafonné à +100**, petit face à la cote d'une issue rare "
                           "(nul/upset à 120-180). Le vrai levier reste donc de trouver le bon **RÉSULTAT rare** ; "
                           "le score exact n'est qu'un bonus. Ne sacrifie pas de proba pour un score absurde "
                           "(au-delà d'« ultra rare » < 0,5 %, le bonus ne monte plus). À très gros écart c'est "
                           "une **loterie** — ce mode maximise ta petite chance, il ne la crée pas.")
        except Exception as e:
            with results_box:
                st.error("❌ Le calcul Jackpot a échoué — détail :")
                st.exception(e)
