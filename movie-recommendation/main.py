from src.data_processing import load_ratings, create_user_item_matrix
from src.similarity import compute_item_similarity
from src.item_based_cf import recommend_movies_item_based
from src.svd_model import recommend_movies_svd

# Load dataset
ratings_df = load_ratings("data/u.data")

# Create user-item matrix
user_item = create_user_item_matrix(ratings_df)

# Item-based CF
item_sim = compute_item_similarity(user_item)
item_cf_recs = recommend_movies_item_based(
    user_id=1,
    user_item_matrix=user_item,
    item_similarity=item_sim
)

print("Item-Based CF Recommendations:")
print(item_cf_recs)

# SVD recommendations
svd_recs = recommend_movies_svd(user_id=1, user_item_matrix=user_item)

print("SVD Recommendations:")
print(svd_recs)