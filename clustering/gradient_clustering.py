import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def compute_cosine_similarity(gradients):
    """
    Compute the cosine similarity matrix between gradient vectors.
    """
    gradients = np.array(gradients)
    sim_matrix = cosine_similarity(gradients)
    return sim_matrix

def cluster_gradients(gradients, n_clusters=2, auto_threshold=False):
    """
    Cluster gradients using Agglomerative Clustering.
    If auto_threshold is True, uses the 25th percentile of distances as threshold.
    Returns: Cluster labels for each gradient.
    """
    pairwise_dist = 1 - cosine_similarity(gradients)
    if auto_threshold:
        threshold = np.percentile(pairwise_dist, 25)
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed',
                                             linkage='average', distance_threshold=threshold)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed',
                                             linkage='average')
    return clustering.fit_predict(pairwise_dist)

def aggregate_gradients_by_cluster(gradients, cluster_labels):
    """
    Aggregate gradients by clustering.
    For each cluster, compute the median of gradients after filtering out outliers,
    then combine using a weighted average.
    Returns: Aggregated gradient vector.
    """
    clusters = defaultdict(list)
    for grad, label in zip(gradients, cluster_labels):
        clusters[label].append(grad)
    
    aggregated = []
    total_weight = 0
    for label, grads in clusters.items():
        grads = np.array(grads)
        q1 = np.percentile(grads, 25, axis=0)
        q3 = np.percentile(grads, 75, axis=0)
        valid = ~np.any((grads < q1) | (grads > q3), axis=1)
        if np.sum(valid) < len(grads) // 4:
            # Ensure at least a quarter of gradients are kept.
            valid[:max(1, len(grads)//4)] = True  
        filtered = grads[valid]
        cluster_median = np.median(filtered, axis=0)
        weight = len(filtered)
        aggregated.append((cluster_median, weight))
        total_weight += weight

    if total_weight == 0:
        raise ValueError("No valid gradients remain after outlier filtering.")
    
    aggregated_gradient = sum(w * m for m, w in aggregated) / total_weight
    return aggregated_gradient
