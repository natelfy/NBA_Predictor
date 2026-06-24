"""
Football Oracle - Signaux de CONTEXTE (talent + fraicheur) pour eclairer les picks
==================================================================================

squad_power.csv / fatigue_index.csv sont des INSTANTANES 2026 -> on les utilise comme
CONTEXTE affiche, ils n'ALTERENT PAS les probabilites calibrees du modele (l'ELO capte
deja la force ; ajouter un tilt non valide casserait la calibration confirmee par le radar).

Usage principal : reperer les matchs a TALENT SERRE -> propices au NUL, qui est le filon
contrarian que le radar montre sous-estime par le modele (P(nul) < frequence reelle).

Pur stdlib (csv, os). ISOLATION : aucun fichier NBA.
"""
import csv
import os


def _load(path, key):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d[r[key]] = r
    return d


def load_signals(squad_path='data/squad_power.csv', fatigue_path='data/fatigue_index.csv'):
    return _load(squad_path, 'Team'), _load(fatigue_path, 'Team')


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def team_context(team, squad_d, fatigue_d):
    sp = _to_float(squad_d.get(team, {}).get('Squad_Power_Index'))
    status = fatigue_d.get(team, {}).get('Physical_Status')
    return {'squad_power': sp, 'fatigue': status, 'known': team in squad_d}


def match_context(t1, t2, squad_d, fatigue_d, close_threshold=4.0, saturation=99.0):
    """
    Contexte des 2 équipes + repère 'nul probable' si talents proches.
    NB: squad_power sature (=99) pour tous les tops -> on NE flagge PAS 'talent serré'
    quand les deux sont saturés (on ne peut pas les départager honnêtement).
    """
    c1 = team_context(t1, squad_d, fatigue_d)
    c2 = team_context(t2, squad_d, fatigue_d)
    diff, draw_prone, note = None, False, ""
    sp1, sp2 = c1['squad_power'], c2['squad_power']
    if sp1 is not None and sp2 is not None:
        diff = sp1 - sp2
        both_saturated = sp1 >= saturation and sp2 >= saturation
        if abs(diff) <= close_threshold and not both_saturated:
            draw_prone = True
            note = "⚖️ Talent serré → match propice au NUL (filon contrarian sous-estimé par le modèle)."
    return {'t1': c1, 't2': c2, 'squad_diff': diff, 'draw_prone': draw_prone, 'note': note}


# ---------------------------------- auto-test ----------------------------------
if __name__ == "__main__":
    sq, fa = load_signals()
    print(f"squad_power: {len(sq)} équipes | fatigue: {len(fa)} équipes")
    for t1, t2 in [("Spain", "Portugal"), ("Brazil", "Curaçao"), ("England", "Netherlands")]:
        mc = match_context(t1, t2, sq, fa)
        sp1 = mc['t1']['squad_power']
        sp2 = mc['t2']['squad_power']
        print(f"\n{t1} vs {t2}")
        print(f"  talent: {sp1} ({mc['t1']['fatigue']}) vs {sp2} ({mc['t2']['fatigue']})")
        print(f"  diff={mc['squad_diff']} draw_prone={mc['draw_prone']} {mc['note']}")
