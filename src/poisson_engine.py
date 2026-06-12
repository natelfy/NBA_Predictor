import pandas as pd
import numpy as np
from scipy.stats import poisson

def calculate_expected_goals(team_a_stats, team_b_stats):
    """
    Extrait les statistiques offensives et défensives pour estimer
    les buts attendus (xG) de chaque équipe dans ce match spécifique.
    """
    # Force offensive (Buts marqués historiquement ajustés par la dynamique)
    off_a = team_a_stats.get('AVG_GOALS_10', 1.5)
    off_b = team_b_stats.get('AVG_GOALS_10', 1.5)
    
    # Force défensive (Buts encaissés historiquement)
    def_a = team_a_stats.get('AVG_OPP_GOALS_10', 1.0)
    def_b = team_b_stats.get('AVG_OPP_GOALS_10', 1.0)
    
    # xG = Force offensive de l'équipe * Force défensive de l'adversaire
    # (Un ratio basique, améliorable avec les vrais xG extraits précédemment si dispo)
    xg_a = off_a * def_b
    xg_b = off_b * def_a
    
    # Ajustement selon l'ELO (la différence de niveau global)
    elo_diff = team_a_stats.get('ELO_POST', 1500) - team_b_stats.get('ELO_POST', 1500)
    
    if elo_diff > 0:
        xg_a += (elo_diff / 400) * 0.5  # L'équipe A est meilleure
        xg_b -= (elo_diff / 400) * 0.2
    else:
        xg_b += (abs(elo_diff) / 400) * 0.5 # L'équipe B est meilleure
        xg_a -= (abs(elo_diff) / 400) * 0.2

    # On s'assure de ne pas avoir de xG négatifs
    return max(0.1, xg_a), max(0.1, xg_b)

def generate_score_matrix(xg_a, xg_b, max_goals=6):
    """
    Génère une matrice de probabilités pour chaque score exact jusqu'à 'max_goals'
    en utilisant la distribution de Poisson.
    """
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            # Loi de Poisson : (lambda^k * e^-lambda) / k!
            prob_a = poisson.pmf(i, xg_a)
            prob_b = poisson.pmf(j, xg_b)
            matrix[i][j] = prob_a * prob_b
            
    return matrix

def get_top_scores(matrix, top_n=3):
    """Extrait les N scores exacts les plus probables de la matrice."""
    scores = []
    max_goals = matrix.shape[0]
    
    for i in range(max_goals):
        for j in range(max_goals):
            scores.append({
                'Score': f"{i}-{j}",
                'Probability': matrix[i][j],
                'Home_Goals': i,
                'Away_Goals': j
            })
            
    df_scores = pd.DataFrame(scores)
    df_scores = df_scores.sort_values('Probability', ascending=False).head(top_n)
    return df_scores