# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings

warnings.simplefilter("always")
warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning)

# FUNCTIONS
def load_data(file_path):
    """Loads the dataset and renames columns."""
    data = pd.read_csv(file_path)
    data.rename(columns={'DATE': 'eksd', 'PATIENT_ID': 'pnr'}, inplace=True)
    return data

def preprocess_data(data):
    """Preprocesses data by converting dates and calculating event intervals."""
    data['eksd'] = pd.to_datetime(data['eksd'])
    data = data.sort_values(by=['pnr', 'eksd'])
    data['prev_eksd'] = data.groupby('pnr')['eksd'].shift(1)
    data = data.dropna()
    data['event_interval'] = (data['eksd'] - data['prev_eksd']).dt.days
    return data[data['event_interval'] > 0]

def ecdf(data):
    """Computes the empirical cumulative distribution function."""
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def plot_ecdf(data):
    """Plots the ECDF for event intervals."""
    x, y = ecdf(data['event_interval'])
    df_ecdf = pd.DataFrame({'x': x, 'y': y})
    df_ecdf = df_ecdf[df_ecdf['y'] <= 0.8]

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_ecdf['x'], df_ecdf['y'], label='80% ECDF', color='tab:blue')  # Changed color
    plt.axhline(y=0.8, linestyle='dashed', color='tab:red', label='80% Threshold')  # Changed color
    plt.xlabel("Event Interval (days)")
    plt.ylabel("ECDF")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, y, label='100% ECDF', color='tab:green')  # Changed color
    plt.xlabel("Event Interval (days)")
    plt.ylabel("ECDF")
    plt.legend()
    
    plt.show()

def plot_density(data):
    """Plots the density estimation for log-transformed event intervals."""
    density = gaussian_kde(np.log(data['event_interval']))
    x_vals = np.linspace(min(np.log(data['event_interval'])), max(np.log(data['event_interval'])), 100)
    y_vals = density(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="Density", color='tab:orange')  # Changed color
    plt.title("Log(Event Interval) Density Plot")
    plt.xlabel("Log(Event Interval)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def kmeans_clustering(data):
    """Performs K-Means clustering and selects the optimal number of clusters using silhouette score."""
    scaled_data = np.log(data[['event_interval']])
    sil_scores = {}

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_data)
        score = silhouette_score(scaled_data, kmeans.labels_)
        sil_scores[k] = score
    
    plot_silhouette_scores(sil_scores)
    
    optimal_k = max(sil_scores, key=sil_scores.get)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(scaled_data)
    data['kmeans_cluster'] = kmeans.labels_
    
    return data

def plot_silhouette_scores(sil_scores):
    """Plots silhouette scores for different cluster sizes."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o', linestyle='-', color='tab:purple')  # Changed color
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.show()

def dbscan_clustering(data):
    """Performs DBSCAN clustering and finds optimal eps value using nearest neighbors."""
    scaled_data = np.log(data[['event_interval']])

    # Determine optimal epsilon
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, indices = neighbors_fit.kneighbors(scaled_data)
    distances = np.sort(distances[:, -1])
    
    eps_value = 0.5  # Default value (adjustable based on data)
    
    dbscan = DBSCAN(eps=eps_value, min_samples=5)
    data['dbscan_cluster'] = dbscan.fit_predict(scaled_data)
    
    return data

def plot_clusters(data):
    """Plots boxplots of event intervals for K-Means and DBSCAN clusters."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=data['kmeans_cluster'], y=data['event_interval'], palette="tab10")  # Changed palette
    plt.axhline(y=np.median(data['event_interval']), linestyle='dashed', color='tab:red', label='Median')  # Changed color
    plt.title("K-Means Clusters")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data['dbscan_cluster'], y=data['event_interval'], palette="tab10")  # Changed palette
    plt.axhline(y=np.median(data['event_interval']), linestyle='dashed', color='tab:red', label='Median')  # Changed color
    plt.title("DBSCAN Clusters")
    
    plt.show()

# MAIN FUNCTION
def main():
    """Main function to run the SEE process."""
    file_path = 'med_events.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    plot_ecdf(data)
    plot_density(data)
    
    data = kmeans_clustering(data)
    data = dbscan_clustering(data)
    
    plot_clusters(data)

if __name__ == "__main__":
    main()