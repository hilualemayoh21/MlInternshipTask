import numpy as np
import pandas as pd

def recommend_movies_item_based(user_id, user_item_matrix, item_similarity, k=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated = user_ratings[user_ratings.isna()].index
    predicted_scores = {}

    for item in unrated:
        similar_items = item_similarity[item]
        rated_similar = similar_items[user_ratings.notna()]
        ratings = user_ratings[rated_similar.index]
        if rated_similar.sum() != 0:
            predicted_scores[item] = np.dot(ratings, rated_similar) / rated_similar.sum()
        else:
            predicted_scores[item] = 0  # fallback if no similar rated items

    recommendations = pd.Series(predicted_scores).sort_values(ascending=False).head(k)
    return recommendations