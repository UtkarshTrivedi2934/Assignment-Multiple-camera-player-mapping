import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment

def match_players(broadcast_feats, tacticam_feats):
    """
    Matches players between broadcast and tacticam using cosine similarity and Hungarian algorithm.
    """

    b_keys = list(broadcast_feats.keys())
    t_keys = list(tacticam_feats.keys())

    # Create feature matrices
    B = np.array([broadcast_feats[k] for k in b_keys])
    T = np.array([tacticam_feats[k] for k in t_keys])

    # Compute distance matrix (cosine distances: lower = better match)
    D = cosine_distances(B, T)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(D)

    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((b_keys[i], t_keys[j], D[i][j]))

    return matches  # List of tuples: (broadcast_key, tacticam_key, distance)
