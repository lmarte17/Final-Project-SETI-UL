import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from .dimensionality_reduction import reduce_dimensions
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

def prepare_anomaly_data(df, signal_sample_size=None):
    """
    Prepare data for anomaly detection with balanced holdout set.
    
    Args:
        df (pd.DataFrame): DataFrame containing image data and labels
        signal_sample_size (int, optional): Number of signal samples to use in holdout set.
            If None, will match noise test size.
            
    Returns:
        tuple: (X_noise_train, X_holdout, y_holdout)
            - X_noise_train: Training data (noise only)
            - X_holdout: Balanced holdout set (noise + signals)
            - y_holdout: Labels for holdout set (0 for noise, 1 for signals)
    """
    # First get the nan mask from reduce_dimensions
    nan_mask = ~df['vertical_profile'].apply(lambda x: np.isnan(x).any())
    df_clean = df[nan_mask]
    
    # Now create noise mask from cleaned dataframe
    noise_mask = df_clean['label'] == 'noise'
    
    # Get feature matrix X_combined from our previous processing
    X_pca, X_umap, labels, X_combined = reduce_dimensions(df)
    
    # Split noise data into train and test
    X_noise = X_combined[noise_mask]
    X_signal = X_combined[~noise_mask]
    
    # If signal_sample_size not specified, match noise test size
    if signal_sample_size is None:
        signal_sample_size = len(X_noise) // 4  # So it matches noise test size
    
    # Randomly sample from signals to match proportions
    signal_indices = np.random.RandomState(42).choice(
        len(X_signal), 
        size=signal_sample_size, 
        replace=False
    )
    X_signal_sampled = X_signal[signal_indices]
    
    # 80/20 split of noise data
    X_noise_train, X_noise_test = train_test_split(
        X_noise, 
        test_size=0.2, 
        random_state=42
    )
    
    # Combine test noise with sampled signals for balanced holdout set
    X_holdout = np.vstack([X_noise_test, X_signal_sampled])
    y_holdout = np.concatenate([
        np.zeros(len(X_noise_test)),   # 0 for noise
        np.ones(len(X_signal_sampled)) # 1 for signals
    ])
    
    print("\nDataset sizes:")
    print(f"Training (noise only): {len(X_noise_train)} samples")
    print(f"Holdout set: {len(X_holdout)} samples")
    print(f"    - Noise test: {len(X_noise_test)} samples")
    print(f"    - Signals: {len(X_signal_sampled)} samples")
    print(f"\nTotal signals available: {len(X_signal)}")
    print(f"Signals used in holdout: {len(X_signal_sampled)}")
    
    return X_noise_train, X_holdout, y_holdout

def detect_anomalies(X_train, X_holdout, y_holdout, contamination=0.1):
    """
    Perform anomaly detection using Isolation Forest.
    
    Args:
        X_train (np.ndarray): Training data (noise only)
        X_holdout (np.ndarray): Holdout set for evaluation
        y_holdout (np.ndarray): Labels for holdout set
        contamination (float): Expected proportion of anomalies in the data
        
    Returns:
        tuple: (iso, auc_score, roc_curve_data)
            - iso: Trained Isolation Forest model
            - auc_score: ROC AUC score on holdout set
            - roc_curve_data: Tuple of (fpr, tpr, thresholds) for ROC curve
    """
    # Train on noise only
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,  # Increased from default
        max_samples='auto',
        bootstrap=True,
        n_jobs=-1
    )
    
    # Fit on training data (noise only)
    iso.fit(X_train)
    
    # Get anomaly scores for holdout set
    holdout_scores = -iso.score_samples(X_holdout)
    
    # Calculate ROC AUC
    auc_score = roc_auc_score(y_holdout, holdout_scores)
    
    # Get ROC curve for plotting
    fpr, tpr, thresholds = roc_curve(y_holdout, holdout_scores)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Score distributions
    noise_scores = holdout_scores[y_holdout == 0]
    signal_scores = holdout_scores[y_holdout == 1]
    ax1.hist(noise_scores, bins=50, alpha=0.5, label='Noise (test)', density=True)
    ax1.hist(signal_scores, bins=50, alpha=0.5, label='Signal', density=True)
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Anomaly Scores (Balanced Holdout Set)')
    ax1.legend()
    
    # Plot 2: ROC curve
    ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')  # diagonal line
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (Balanced Evaluation)')
    ax2.legend()
    
    plt.tight_layout()
    
    print(f"\nROC AUC Score on balanced holdout set: {auc_score:.3f}")
    
    return iso, auc_score, (fpr, tpr, thresholds)

def enhanced_anomaly_detection(df, n_components=10):
    """
    Anomaly detection using both SVD components and original features.
    
    Args:
        df (pd.DataFrame): DataFrame containing image data and labels
        n_components (int): Number of components to extract for SVD/NMF
        
    Returns:
        tuple: (iso, svd, nmf, auc_score)
            - iso: Trained Isolation Forest model
            - svd: Fitted SVD model
            - nmf: Fitted NMF model
            - auc_score: ROC AUC score on holdout set
    """
    # 1. Get SVD/NMF components
    X_img = np.vstack(df['flattened'])
    
    # SVD transformation
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_img)
    
    # NMF transformation (ensure non-negative input)
    X_img_pos = X_img - X_img.min()
    nmf = NMF(n_components=n_components, random_state=42)
    X_nmf = nmf.fit_transform(X_img_pos)
    
    # 2. Calculate reconstruction error
    svd_reconstruction = svd.inverse_transform(X_svd)
    nmf_reconstruction = np.dot(X_nmf, nmf.components_)
    
    # Compute reconstruction errors for each sample
    svd_errors = np.mean((X_img - svd_reconstruction) ** 2, axis=1)
    nmf_errors = np.mean((X_img_pos - nmf_reconstruction) ** 2, axis=1)
    
    # 3. Combine features
    X_combined = np.column_stack([
        X_svd,                    # SVD components
        X_nmf,                    # NMF components
        svd_errors[:, np.newaxis],# SVD reconstruction error
        nmf_errors[:, np.newaxis] # NMF reconstruction error
    ])
    
    # 4. Split data for anomaly detection
    noise_mask = df['label'] == 'noise'
    
    # Get indices for noise samples
    noise_indices = np.where(noise_mask)[0]
    signal_indices = np.where(~noise_mask)[0]
    
    # Split noise into train/test
    noise_train_idx, noise_test_idx = train_test_split(
        noise_indices, test_size=0.2, random_state=42
    )
    
    # Sample from signals to match test size
    n_test_signals = len(noise_test_idx)
    signal_test_idx = np.random.choice(
        signal_indices, size=n_test_signals, replace=False
    )
    
    # Create training and test sets
    X_train = X_combined[noise_train_idx]
    X_holdout = np.vstack([
        X_combined[noise_test_idx],
        X_combined[signal_test_idx]
    ])
    y_holdout = np.concatenate([
        np.zeros(len(noise_test_idx)),
        np.ones(len(signal_test_idx))
    ])
    
    # 5. Train Isolation Forest with enhanced features
    iso = IsolationForest(
        contamination=0.1,
        n_estimators=200,      # Increase estimators
        max_samples=256,       # Adjust sample size
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    iso.fit(X_train)
    holdout_scores = -iso.score_samples(X_holdout)
    
    # 6. Evaluate and visualize results
    auc_score = roc_auc_score(y_holdout, holdout_scores)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot score distributions
    noise_scores = holdout_scores[y_holdout == 0]
    signal_scores = holdout_scores[y_holdout == 1]
    ax1.hist(noise_scores, bins=50, alpha=0.5, label='Noise', density=True)
    ax1.hist(signal_scores, bins=50, alpha=0.5, label='Signal', density=True)
    ax1.set_title('Anomaly Score Distribution')
    ax1.legend()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_holdout, holdout_scores)
    ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve')
    ax2.legend()
    
    # Plot reconstruction errors
    ax3.scatter(svd_errors[noise_mask], nmf_errors[noise_mask], 
               alpha=0.5, label='Noise')
    ax3.scatter(svd_errors[~noise_mask], nmf_errors[~noise_mask], 
               alpha=0.5, label='Signal')
    ax3.set_xlabel('SVD Reconstruction Error')
    ax3.set_ylabel('NMF Reconstruction Error')
    ax3.set_title('Reconstruction Errors')
    ax3.legend()
    
    plt.tight_layout()
    
    print(f"\nROC AUC Score with enhanced features: {auc_score:.3f}")
    
    return iso, svd, nmf, auc_score, fig 