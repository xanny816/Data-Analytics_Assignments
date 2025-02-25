import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.simplefilter("always")
warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning)

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

def plot_ecdf(data):
    """Plots the ECDF for event intervals."""
    x = np.sort(data['event_interval'])
    y = np.arange(1, len(x) + 1) / len(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='ECDF', color='tab:blue')
    plt.axhline(y=0.8, linestyle='dashed', color='tab:red', label='80% Threshold')
    plt.xlabel("Event Interval (days)")
    plt.ylabel("ECDF")
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
    
    optimal_k = max(sil_scores, key=sil_scores.get)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(scaled_data)
    data['kmeans_cluster'] = kmeans.labels_
    
    plot_silhouette_scores(sil_scores)
    return data

def plot_silhouette_scores(sil_scores):
    """Plots silhouette scores for different cluster sizes."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o', linestyle='-', color='tab:purple')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.show()

def plot_kmeans_clusters(data):
    """Plots boxplots of event intervals for K-Means clusters."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data['kmeans_cluster'], y=data['event_interval'], palette="tab10")
    plt.axhline(y=np.median(data['event_interval']), linestyle='dashed', color='tab:red', label='Median')
    plt.title("K-Means Clusters")
    plt.legend()
    plt.show()

def main():
    """Main function to run the K-Means clustering process."""
    file_path = 'med_events.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    plot_ecdf(data)
    data = kmeans_clustering(data)
    plot_kmeans_clusters(data)

if __name__ == "__main__":
    main()
