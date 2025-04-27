import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF

def analyze_matrix_factorization(df, n_components=10):
    """
    Analyze spectrograms using SVD and NMF to find underlying patterns.
    
    Args:
        df (pd.DataFrame): DataFrame containing flattened image data
        n_components (int): Number of components to extract
        
    Returns:
        tuple: (svd, nmf, X_svd, X_nmf)
            - svd: Fitted TruncatedSVD model
            - nmf: Fitted NMF model
            - X_svd: SVD transformed data
            - X_nmf: NMF transformed data
    """
    # Get flattened image data
    X_img = np.vstack(df['flattened'])
    
    # Reshape to ensure we have (n_samples, n_pixels)
    n_samples = X_img.shape[0]
    img_size = int(np.sqrt(X_img.shape[1]))  # Should be 128
    
    print(f"Data shape: {X_img.shape}")
    print(f"Image size: {img_size}x{img_size}")
    
    # Perform SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_img)
    
    # Perform NMF (ensure non-negative input)
    X_img_pos = X_img - X_img.min()  # Make all values non-negative
    nmf = NMF(n_components=n_components, random_state=42)
    X_nmf = nmf.fit_transform(X_img_pos)
    
    # Plot components
    fig1, axes = plt.subplots(2, n_components, figsize=(20, 5))
    
    # Plot SVD components
    for i in range(n_components):
        component = svd.components_[i].reshape(img_size, img_size)
        axes[0, i].imshow(component, cmap='viridis')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'SVD {i+1}')
    
    # Plot NMF components
    for i in range(n_components):
        component = nmf.components_[i].reshape(img_size, img_size)
        axes[1, i].imshow(component, cmap='viridis')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'NMF {i+1}')
    
    plt.suptitle('SVD vs NMF Components', y=1.02)
    plt.tight_layout()
    
    # Plot explained variance for SVD
    fig2, ax = plt.subplots(figsize=(10, 4))
    explained_var = svd.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    ax.plot(range(1, n_components + 1), cumulative_var, 'bo-')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title('SVD Explained Variance')
    ax.grid(True)
    
    # Print explained variance information
    print("\nSVD Explained Variance:")
    print(f"Total variance explained by {n_components} components: {cumulative_var[-1]:.3f}")
    
    # Analyze reconstruction
    def plot_reconstruction(original, reconstructed, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.imshow(original.reshape(img_size, img_size), cmap='viridis')
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(reconstructed.reshape(img_size, img_size), cmap='viridis')
        ax2.set_title('Reconstructed')
        ax2.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    # Show example reconstructions
    idx = 0  # First image as example
    
    # SVD reconstruction
    svd_reconstructed = np.dot(X_svd[idx], svd.components_)
    fig3 = plot_reconstruction(X_img[idx], svd_reconstructed, 'SVD Reconstruction')
    
    # NMF reconstruction
    nmf_reconstructed = np.dot(X_nmf[idx], nmf.components_)
    fig4 = plot_reconstruction(X_img_pos[idx], nmf_reconstructed, 'NMF Reconstruction')
    
    return svd, nmf, X_svd, X_nmf, (fig1, fig2, fig3, fig4) 