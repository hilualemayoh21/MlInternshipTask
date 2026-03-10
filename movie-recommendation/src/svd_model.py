import numpy as np
import pandas as pd
from numpy.linalg import svd

def svd_reconstruct(user_item_matrix):
    R = user_item_matrix.fillna(0).values
    U, sigma, Vt = svd(R, full_matrices=False)
    Sigma = np.diag(sigma)
    R_pred = np.dot(np.dot(U, Sigma), Vt)
    return pd.DataFrame(R_pred, index=user_item_matrix.index, columns=user_item_matrix.columns)

def recommend_movies_svd(user_id, user_item_matrix, k=5):
    R_pred = svd_reconstruct(user_item_matrix)
    user_ratings = user_item_matrix.loc[user_id]
    unrated = user_ratings[user_ratings.isna()].index
    recommendations = R_pred.loc[user_id, unrated].sort_values(ascending=False).head(k)
    return recommendations