import matplotlib.pyplot as plt


def plot_elbow(wcss):
    """
    Plot elbow curve
    """

    plt.plot(range(1, 11), wcss, marker='o')

    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.show()


def plot_clusters(df):
    """
    Visualize clusters
    """

    plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster']
    )

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments")

    plt.show()
def plot_dbscan(df):
    plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['DBSCAN_Cluster']
    )

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("DBSCAN Customer Clusters")

    plt.show()
        
