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


clf, reg, features_list, history_df, ratings = load_oracle_environment()
upcoming_matches = load_upcoming_matches()
squad_d, fatigue_d = load_context_signals()
if clf is None:
    st.stop()


# =====================================================================
# SIDEBAR : une seule entrée essentielle
# =====================================================================
st.sidebar.header("🎯 Ta situation")
gap_3e = st.sidebar.number_input("Points à combler sur le 3e", value=505, step=10)
st.sidebar.caption(f"team_ratings.json : {'✅ chargé (scores ajustés)' if ratings else '❌ absent (fallback goal_diff)'}")

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

                slate.append(mpp_sim.build_match(            # bonus via score_model.KNOWN_BONUS
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

            rec = mpp_sim.recommend(slate, gap=gap_3e, trials=TRIALS, competitor=COMPETITOR)

            with results_box:
                p = rec['p_top3']
                st.markdown("## 🧮 Stratégie recommandée")
                cA, cB = st.columns([1, 2])
                cA.metric("P(combler l'écart au 3e)", f"{p:.1%}")
                cB.info(f"Meilleure stratégie : **{rec['best']}**. "
                        + ("⚠️ Long shot honnête : vise les coups contrarian ci-dessous et relance chaque jour."
                           if p < 0.10 else "Profil jouable : tiens cette ligne et relance chaque jour."))
                st.caption("📌 Radar live : le modèle **sous-estime les nuls** — sur un match à talent serré "
                           "ou face à un gros favori, le **NUL** est un meilleur pari contrarian que l'EV ne le "
                           "montre (voir colonnes *Talent* / *Contexte* dans le détail).")

                st.markdown("### ✅ Tes picks du jour")
                for d in rec['picks']:
                    name = {'T1': d['match'].split(' vs ')[0], 'T2': d['match'].split(' vs ')[1],
                            'NUL': 'MATCH NUL'}[d['pick']]
                    sc = f"{d['score'][0]}-{d['score'][1]}"
                    if d['contrarian']:
                        st.success(f"🎯 **{d['match']}** → **{name}** (score {sc}) · "
                                   f"modèle {d['p_model']:.0%} vs foule {d['p_crowd']:.0%} "
                                   f"(**edge {d['edge'] * 100:+.0f}**, cote {d['cote']}) — COUP CONTRARIAN")
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
                st.caption("On ne grimpe qu'en ayant raison là où la foule a tort (edge>0). "
                           "x2 déjà dépensé. Relance ce calcul à chaque journée.")
        except Exception as e:
            with results_box:
                st.error("❌ Le calcul a échoué — détail ci-dessous (copie-le moi pour que je corrige).")
                st.exception(e)
