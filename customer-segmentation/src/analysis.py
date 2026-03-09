def analyze_clusters(df):
    """
    Compute statistics per cluster
    """

    result = df.groupby("Cluster").mean()

    print("\nCluster Analysis")
    print(result)

    return result