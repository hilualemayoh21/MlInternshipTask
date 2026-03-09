from src.load_data import load_dataset
from src.preprocessing import select_features, scale_features
from src.clustering import find_optimal_clusters, train_kmeans
from src.visualization import plot_elbow, plot_clusters
from src.analysis import analyze_clusters
from src.dbscan_model import run_dbscan


def main():

    print("Step 1: Loading dataset...")

    df = load_dataset("data/Mall_Customers.csv")

    print("Step 2: Selecting features...")

    X = select_features(df)

    print("Step 3: Scaling features...")

    X_scaled = scale_features(X)

    print("Step 4: Finding optimal clusters...")

    wcss = find_optimal_clusters(X_scaled)

    plot_elbow(wcss)

    print("Step 5: Training KMeans...")

    model, labels = train_kmeans(X_scaled, 5)

    df["Cluster"] = labels

    print("Step 6: Visualizing clusters...")

    plot_clusters(df)

    print("Step 7: Cluster analysis...")

    analyze_clusters(df)

    print("Step 8: Running DBSCAN...")

    dbscan_labels = run_dbscan(X_scaled)

    df["DBSCAN_Cluster"] = dbscan_labels

    print(df["DBSCAN_Cluster"].value_counts())


if __name__ == "__main__":
    main()