import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_dimensions(df):
    """
    Perform dimensionality reduction on image data, profiles, and statistical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing image data, profiles, and statistical features
        
    Returns:
        tuple: (X_pca, X_umap, labels, X_combined)
            - X_pca: 2D PCA projection of the data
            - X_umap: 2D UMAP projection of the data
            - labels: Cleaned labels from the input DataFrame
            - X_combined: Combined feature matrix before final dimensionality reduction
    """
    # Find and remove rows with NaN profiles
    nan_mask = ~df['vertical_profile'].apply(lambda x: np.isnan(x).any())
    df_clean = df[nan_mask]
    
    # Process image data
    X_img = np.vstack(df_clean['flattened'])
    X_img_scaled = StandardScaler().fit_transform(X_img)
    pca_img = PCA(n_components=50).fit_transform(X_img_scaled)
    
    # Process profiles
    X_profiles = np.column_stack([
        np.vstack(df_clean['vertical_profile']),
        np.vstack(df_clean['horizontal_profile'])
    ])
    X_profiles_scaled = StandardScaler().fit_transform(X_profiles)
    pca_profiles = PCA(n_components=10).fit_transform(X_profiles_scaled)
    
    # Process statistical features
    stat_features = ['max', 'min', 'mean', 'std', 'skewness', 'kurtosis']
    X_stats = df_clean[stat_features].values
    X_stats_scaled = StandardScaler().fit_transform(X_stats)
    
    # Combine all features
    X_combined = np.column_stack([
        pca_img,          # 50 components
        pca_profiles,     # 10 components
        X_stats_scaled    # 6 components
    ])
    
    # Final dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_combined)
    
    return X_pca, X_umap, df_clean['label'], X_combined

def plot_reduced_dimensions(X_pca, X_umap, labels):
    """
    Plot PCA and UMAP projections of the reduced data.
    
    Args:
        X_pca (np.ndarray): 2D PCA projection of the data
        X_umap (np.ndarray): 2D UMAP projection of the data
        labels (np.ndarray): Class labels for the data points
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot PCA
    for label in np.unique(labels):
        mask = labels == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6)
    ax1.set_title('PCA Projection')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot UMAP
    for label in np.unique(labels):
        mask = labels == label
        ax2.scatter(X_umap[mask, 0], X_umap[mask, 1], label=label, alpha=0.6)
    ax2.set_title('UMAP Projection')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig 