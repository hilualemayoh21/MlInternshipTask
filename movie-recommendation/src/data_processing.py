import pandas as pd

def load_ratings(ratings_path='data/u.data'):
    # MovieLens u.data format: user_id, item_id, rating, timestamp
    df = pd.read_csv(ratings_path, sep='\t', names=['user_id','item_id','rating','timestamp'])
    return df

def create_user_item_matrix(df):
    user_item = df.pivot(index='user_id', columns='item_id', values='rating')
    return user_item