import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report

def data_set_loading(path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
        Loading function
    """
    df = pd.read_csv(path)

    startindices = df.index[df["Cycle_Index"] == 1.0].tolist()

    battery_segments = []
    for i in range(len(startindices)):
        start = startindices[i]
        end = startindices[i + 1] if i + 1 < len(startindices) else len(df)
        battery_segments.append(df.iloc[start:end].reset_index(drop=True))

    num_batteries = len(battery_segments)
    num_train = int(num_batteries * 0.8)

    train_segments = battery_segments[:num_train]
    test_segments = battery_segments[num_train:]

    return train_segments, test_segments

def get_fixed_length_rul_curves(segments, n_points=100):
    curves = []
    for seg in segments:
        x = seg["Cycle_Index"].values
        y = seg["RUL"].values
        x_norm = (x - x.min()) / (x.max() - x.min())
        f = interp1d(x_norm, y, kind='linear')
        x_sampled = np.linspace(0, 1, n_points)
        y_sampled = f(x_sampled)
        curves.append(y_sampled)
    return np.vstack(curves)

def prepare_temporal_data(segments):
    merged = []
    segment_lengths = []
    for df in segments:
        cycle_idx = df["Cycle_Index"].values / df["Cycle_Index"].max()
        rul_diff = np.concatenate(([0], np.diff(df["RUL"].values)))
        for vec, cycle, diff in zip(df.to_numpy(), cycle_idx, rul_diff):
            merged.append(np.concatenate([vec, [cycle], [diff]]))
        segment_lengths.append(len(df))
    return np.array(merged), segment_lengths

train_segments, test_segments = data_set_loading("./Battery_RUL.csv")
all_segments = train_segments + test_segments

rul_curves = get_fixed_length_rul_curves(all_segments, n_points=100)
rul_curves_scaled = StandardScaler().fit_transform(rul_curves)

kmeans = KMeans(n_clusters=3, random_state=42)
segment_labels = kmeans.fit_predict(rul_curves_scaled)

X, segment_lengths = prepare_temporal_data(all_segments)
X_scaled = StandardScaler().fit_transform(X)

cycle_labels = []
start_idx = 0
for label, length in zip(segment_labels, segment_lengths):
    cycle_labels.extend([label] * length)
    start_idx += length
cycle_labels = np.array(cycle_labels)

normal_mask = (cycle_labels == 2)  
X_normal = X_scaled[normal_mask]
svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)  
svm.fit(X_normal)

anomaly_scores = svm.predict(X_scaled)
anomalies = anomaly_scores == -1  

pseudo_labels = np.where(cycle_labels == 2, 1, -1)

conf_matrix = confusion_matrix(pseudo_labels, anomaly_scores)
print("Confusion Matrix:")
print(conf_matrix)

target_names = ['Anomaly', 'Normal']
print("\nClassification Report:")
print(classification_report(pseudo_labels, anomaly_scores, target_names=target_names, labels=[-1, 1]))

normal_points = X_scaled[~anomalies]
normal_labels = cycle_labels[~anomalies]
if len(np.unique(normal_labels)) > 1:  
    silhouette_avg = silhouette_score(normal_points, normal_labels)
    print(f"\nSilhouette Score for normal points: {silhouette_avg:.3f}")

np.random.seed(42)
random_labels = np.random.choice([-1, 1], size=len(anomaly_scores), p=[0.1, 0.9])  
random_conf_matrix = confusion_matrix(pseudo_labels, random_labels)
print("\nRandom Baseline Confusion Matrix:")
print(random_conf_matrix)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(X_scaled)

plt.scatter(X_embedded[~anomalies, 0], X_embedded[~anomalies, 1], c='blue', label='Normal', alpha=0.6)
plt.scatter(X_embedded[anomalies, 0], X_embedded[anomalies, 1], c='red', label='Anomaly', alpha=0.6)
for i, label in enumerate(np.unique(cycle_labels)):
    mask = (cycle_labels == label)
    plt.scatter([], [], c=plt.cm.tab10(i), label=f'Cluster {label}', alpha=0.6)  

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Anomaly Detection Using Cluster 2 as Normal')
plt.legend()
plt.show()

print(f"\nAnomaly distribution by cluster: {[(l, np.sum(anomalies[cycle_labels == l]), np.mean(anomalies[cycle_labels == l])) for l in np.unique(cycle_labels)]}")