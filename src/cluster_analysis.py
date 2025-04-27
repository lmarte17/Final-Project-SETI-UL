import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def analyze_clusters(X_pca, X_umap, true_labels):
    """
    Analyze clustering on both PCA and UMAP projections using K-means.
    
    Args:
        X_pca (np.ndarray): 2D PCA projection of the data
        X_umap (np.ndarray): 2D UMAP projection of the data
        true_labels (np.ndarray): True class labels for the data points
        
    Returns:
        tuple: (clusters_pca, clusters_umap)
            - clusters_pca: K-means cluster assignments for PCA projection
            - clusters_umap: K-means cluster assignments for UMAP projection
    """
    # Number of known classes
    n_clusters = len(np.unique(true_labels))
    
    # Fit K-means on both projections
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Get cluster assignments
    clusters_pca = kmeans_pca.fit_predict(X_pca)
    clusters_umap = kmeans_umap.fit_predict(X_umap)
    
    # Calculate silhouette scores
    sil_pca = silhouette_score(X_pca, clusters_pca)
    sil_umap = silhouette_score(X_umap, clusters_umap)
    
    print(f"Silhouette score (PCA): {sil_pca:.3f}")
    print(f"Silhouette score (UMAP): {sil_umap:.3f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Original labels
    for label in np.unique(true_labels):
        mask = true_labels == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6)
    ax1.set_title('PCA - True Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for label in np.unique(true_labels):
        mask = true_labels == label
        ax2.scatter(X_umap[mask, 0], X_umap[mask, 1], label=label, alpha=0.6)
    ax2.set_title('UMAP - True Labels')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # K-means clusters
    scatter1 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, cmap='tab10', alpha=0.6)
    ax3.set_title('PCA - K-means Clusters')
    plt.colorbar(scatter1, ax=ax3)
    
    scatter2 = ax4.scatter(X_umap[:, 0], X_umap[:, 1], c=clusters_umap, cmap='tab10', alpha=0.6)
    ax4.set_title('UMAP - K-means Clusters')
    plt.colorbar(scatter2, ax=ax4)
    
    plt.tight_layout()
    return clusters_pca, clusters_umap

def analyze_feature_importance(X_combined, pca, feature_names=None):
    """
    Analyze which features contribute most to the clustering.
    
    Args:
        X_combined (np.ndarray): Combined feature matrix
        pca (sklearn.decomposition.PCA): Fitted PCA object
        feature_names (list, optional): Names of features. If None, generic names will be used.
        
    Returns:
        pd.DataFrame: DataFrame containing feature importance information
    """
    # Get PCA components importance
    explained_var_ratio = pca.explained_variance_ratio_
    
    # Get feature contributions to first two PCs
    components = pca.components_[:2]  # Get first two components
    
    # If feature names not provided, create generic ones
    if feature_names is None:
        feature_names = [
            *[f'img_{i}' for i in range(50)],          # PCA image features
            *[f'profile_{i}' for i in range(10)],      # PCA profile features
            'max', 'min', 'mean', 'std', 'skewness', 'kurtosis'  # Statistical features
        ]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'PC1': components[0],
        'PC2': components[1],
        'Feature': feature_names
    })
    
    # Calculate absolute contribution
    importance_df['Total_Contribution'] = np.sqrt(importance_df['PC1']**2 + importance_df['PC2']**2)
    
    # Sort by total contribution
    importance_df = importance_df.sort_values('Total_Contribution', ascending=False)
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
    ax1.set_title('Explained Variance Ratio by Component')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    
    # Plot top feature contributions
    top_n = 10
    ax2.barh(importance_df['Feature'][:top_n], importance_df['Total_Contribution'][:top_n])
    ax2.set_title(f'Top {top_n} Feature Contributions')
    ax2.set_xlabel('Contribution Magnitude')
    
    plt.tight_layout()
    
    # Print detailed information about top features
    print("\nTop 10 Most Important Features:")
    print(importance_df[['Feature', 'Total_Contribution']].head(10))
    
    # Group contribution by feature type
    feature_types = {
        'Image Features': importance_df[importance_df['Feature'].str.startswith('img_')]['Total_Contribution'].sum(),
        'Profile Features': importance_df[importance_df['Feature'].str.startswith('profile_')]['Total_Contribution'].sum(),
        'Statistical Features': importance_df[~importance_df['Feature'].str.contains('_')]['Total_Contribution'].sum()
    }
    
    print("\nContribution by Feature Type:")
    for feat_type, contribution in feature_types.items():
        print(f"{feat_type}: {contribution:.3f}")
    
    return fig, importance_df 