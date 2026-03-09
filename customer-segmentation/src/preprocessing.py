from sklearn.preprocessing import StandardScaler


def select_features(df):
    """
    Select the features used for clustering
    """

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    return X


def scale_features(X):
    """
    Scale features using StandardScaler
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled