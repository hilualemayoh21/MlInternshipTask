def precision_at_k(recommended_items, actual_items, k=5):
    recommended_top_k = recommended_items.head(k).index
    hits = len(set(recommended_top_k) & set(actual_items))
    return hits / k