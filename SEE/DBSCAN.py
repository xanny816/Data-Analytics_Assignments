import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.simplefilter("always")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

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

def plot_density(data):
    """Plots the density estimation for log-transformed event intervals."""
    density = gaussian_kde(np.log(data['event_interval']))
    x_vals = np.linspace(min(np.log(data['event_interval'])), max(np.log(data['event_interval'])), 100)
    y_vals = density(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="Density", color='tab:orange')
    plt.title("Log(Event Interval) Density Plot")
    plt.xlabel("Log(Event Interval)")
    plt.ylabel("Density")
    plt.legend()
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
    """Plots boxplot of event intervals for DBSCAN clusters."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data['dbscan_cluster'], y=data['event_interval'], palette="tab10")
    plt.axhline(y=np.median(data['event_interval']), linestyle='dashed', color='tab:red', label='Median')
    plt.title("DBSCAN Clusters")
    plt.legend()
    plt.show()

def main():
    """Main function to run DBSCAN clustering process."""
    file_path = 'med_events.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    plot_density(data)
    data = dbscan_clustering(data)
    plot_clusters(data)

if __name__ == "__main__":
    main()
