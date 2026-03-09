from sklearn.cluster import DBSCAN


def run_dbscan(X_scaled):

    dbscan = DBSCAN(eps=0.5, min_samples=5)

    labels = dbscan.fit_predict(X_scaled)

    return labels