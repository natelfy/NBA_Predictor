"""
Football Oracle - Back-test MODELE vs FOULE (le juge de paix)
=============================================================

Question : "Mon modele bat-il vraiment la foule de ~599 joueurs, et SUR QUELS
matchs ?" -> seul moyen de savoir ou se differencier (edge reel) vs ou suivre la
foule. Compare, sur les matchs WC deja joues, la log loss / Brier / accuracy du
MODELE (data/mpp_live_tracking.csv) contre celles de la FOULE (% que tu fournis).

Usage :
  1) python src/backtest_vs_crowd.py --template
       -> ecrit data/crowd_history_TEMPLATE.csv (date,home,away + colonnes % vides),
          pre-rempli avec TES matchs joues. Colle tes % de foule, enregistre sous
          data/crowd_history.csv.
  2) python src/backtest_vs_crowd.py
       -> verdict global + par segment (foule dominante vs match ouvert, + nuls).

Format crowd_history.csv : date,home,away,crowd_home_pct,crowd_draw_pct,crowd_away_pct
(les % peuvent etre 0-100 ou 0-1 ; renormalises automatiquement).

Pur stdlib. ISOLATION : aucun fichier NBA.
"""
import csv
import os
import sys
import math
import unicodedata

TRACK = 'data/mpp_live_tracking.csv'
CROWD = 'data/crowd_history.csv'
TEMPLATE = 'data/crowd_history_TEMPLATE.csv'
EPS = 1e-15
DOMINANT = 0.78


def _norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().strip()
    return ' '.join(s.split())


def _key(date, home, away):
    return (str(date).strip(), _norm(home), _norm(away))


def read_tracking(path=TRACK):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            home, _, away = r['MATCH'].partition(' vs ')
            rows[_key(r['DATE'], home, away)] = {
                'date': r['DATE'], 'home': home.strip(), 'away': away.strip(),
                'p_model': {'T1': float(r['PROB_T1']), 'NUL': float(r['PROB_NUL']), 'T2': float(r['PROB_T2'])},
                'act': int(float(r['ACTUAL_CLASS'])),   # 0=T2, 1=NUL, 2=T1
            }
    return rows


def _pct(v):
    return float(v) if v not in (None, '') else 0.0


def read_crowd(path=CROWD):
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            h = _pct(r.get('crowd_home_pct'))
            d = _pct(r.get('crowd_draw_pct'))
            a = _pct(r.get('crowd_away_pct'))
            s = h + d + a
            if s <= 0:
                continue
            out[_key(r['date'], r['home'], r['away'])] = {'T1': h / s, 'NUL': d / s, 'T2': a / s}
    return out


def make_template(track_path=TRACK, out=TEMPLATE):
    rows = read_tracking(track_path)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['date', 'home', 'away', 'crowd_home_pct', 'crowd_draw_pct', 'crowd_away_pct'])
        for v in sorted(rows.values(), key=lambda x: x['date']):
            w.writerow([v['date'], v['home'], v['away'], '', '', ''])
    print(f"✅ Template écrit : {out} ({len(rows)} matchs). Colle tes % de foule, "
          f"enregistre sous {CROWD}, puis relance sans --template.")


def _metrics(recs, prob_key):
    n = len(recs)
    cls = {0: 'T2', 1: 'NUL', 2: 'T1'}
    ll = -sum(math.log(max(r[prob_key][cls[r['act']]], EPS)) for r in recs) / n
    acc = sum(1 for r in recs if max(('T1', 'NUL', 'T2'), key=lambda k: r[prob_key][k]) == cls[r['act']]) / n
    brier = sum(sum((r[prob_key][cls[k]] - (1.0 if r['act'] == k else 0.0)) ** 2 for k in (0, 1, 2))
                for r in recs) / n / 3.0
    return {'n': n, 'll': ll, 'acc': acc, 'brier': brier}


def _verdict(m, c):
    if m['ll'] < c['ll'] - 0.01:
        return "✅ le MODÈLE bat la foule → DIFFÉRENCIE-toi ici"
    if c['ll'] < m['ll'] - 0.01:
        return "❌ la FOULE bat le modèle → suis la foule ici"
    return "≈ égalité → pas d'edge net"


def _print_block(title, recs):
    if not recs:
        print(f"\n[{title}] aucun match.")
        return
    m, c = _metrics(recs, 'p_model'), _metrics(recs, 'p_crowd')
    print(f"\n[{title}] n={m['n']}")
    print(f"   log loss : modèle {m['ll']:.4f}  |  foule {c['ll']:.4f}   -> {_verdict(m, c)}")
    print(f"   accuracy : modèle {m['acc']:.0%}     |  foule {c['acc']:.0%}")
    print(f"   Brier    : modèle {m['brier']:.4f} |  foule {c['brier']:.4f}")


def backtest(track_path=TRACK, crowd_path=CROWD):
    track = read_tracking(track_path)
    crowd = read_crowd(crowd_path)
    recs, missing = [], []
    for k, v in track.items():
        if k in crowd:
            v['p_crowd'] = crowd[k]
            recs.append(v)
        else:
            missing.append(v)
    print("=" * 64)
    print(f"⚖️  BACK-TEST MODÈLE vs FOULE — {len(recs)} matchs appariés"
          + (f" ({len(missing)} sans % foule)" if missing else ""))
    print("=" * 64)
    if not recs:
        print("Aucun match apparié. Vérifie les noms/dates dans crowd_history.csv.")
        if missing:
            print("Exemples attendus :", [f"{v['date']} {v['home']} vs {v['away']}" for v in missing[:3]])
        return

    _print_block("GLOBAL", recs)
    _print_block("Foule DOMINANTE (max ≥ 78%)", [r for r in recs if max(r['p_crowd'].values()) >= DOMINANT])
    _print_block("Matchs OUVERTS (max < 78%)", [r for r in recs if max(r['p_crowd'].values()) < DOMINANT])
    _print_block("Matchs nuls (résultat réel = NUL)", [r for r in recs if r['act'] == 1])

    print("\n" + "-" * 64)
    print("Lecture : différencie-toi UNIQUEMENT sur les segments où le modèle bat la foule.")
    print("Si la foule gagne partout -> suivre la foule (et accepter que le top-3 = chance).")
    print("=" * 64)


if __name__ == "__main__":
    if '--template' in sys.argv:
        make_template()
    elif not os.path.exists(CROWD):
        print(f"❌ {CROWD} introuvable. Lance d'abord :  python src/backtest_vs_crowd.py --template")
    else:
        backtest()
