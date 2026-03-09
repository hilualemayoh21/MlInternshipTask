from sklearn.cluster import KMeans


def find_optimal_clusters(X_scaled):
    """
    Compute WCSS for elbow method
    """

    wcss = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)

        wcss.append(kmeans.inertia_)

    return wcss


def train_kmeans(X_scaled, n_clusters=5):
    """
    Train KMeans clustering model
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    labels = kmeans.fit_predict(X_scaled)

    return kmeans, labels