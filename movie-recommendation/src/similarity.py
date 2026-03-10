import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_user_similarity(user_item_matrix):
    return pd.DataFrame(
        cosine_similarity(user_item_matrix.fillna(0)),
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

def compute_item_similarity(user_item_matrix):
    return pd.DataFrame(
        cosine_similarity(user_item_matrix.fillna(0).T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )