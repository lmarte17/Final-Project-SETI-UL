# Final-Project-SETI-UL
# SETI Signal Classification Using Unsupervised Learning

## Project Overview
This project attempts to develop an unsupervised learning approach to SETI (Search for Extraterrestrial Intelligence) signal classification. It addresses a key challenge in SETI research: efficiently categorizing the vast amounts of signal data collected from radio telescopes, which traditionally requires manual review by researchers. The overall goal of this project is to accurately and reliably detect anomalies (signals) against normal (noise) spectrograms.

The project analyzes a dataset of 7,000 spectrograms containing seven distinct types of signals commonly encountered in SETI research, ranging from narrowband transmissions to squiggle patterns.



## Technical Approach
The project combines advanced image processing, feature engineering, and multiple unsupervised learning techniques:

### 1. Data and Preprocessing Pipeline
- Data Source: Spectogram image dataset: [SETI-Dataset](https://www.kaggle.com/datasets/tentotheminus9/seti-data)
    - Noise
    - Six types of signals
- Preprocessing:
    - Statistical Filtering: Clipped pixel intensities to emphasize significant signals.
    - Gaussian Blurring: Reduced noise, emphasizing prominent features.
    - Morphological Operations: Removed small noise artifacts.
    - Edge Detection (Sobel): Highlighted prominent edges indicating signal presence.
    - Intensity Profiles and Gradients: Derived informative summaries of each spectrogram image.
    - Feature Normalization: Ensured consistency across samples.
- Signal profile analysis

### 2. Dimensionality Reduction and Feature Engineering
Multiple methods were applied to reduce dimensionality and extract robust features from the processed images:
- Principal Component Analysis (PCA):
    - Linear dimensionality reduction, capturing global variance.
    - Effective but limited by linear assumptions.
- UMAP (Uniform Manifold Approximation and Projection):
    - Non-linear reduction capturing complex data structures.
    - Clearly visualized and revealed distinct clusters of signal types.
- Matrix Factorization Techniques:
    - Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF) extracted underlying image patterns.
    - Both SVD and NMF produced reconstruction errors utilized as strong anomaly indicators.
- Combined Feature Set:
    - Integrated embeddings, reconstruction errors, and statistical descriptors into a comprehensive feature set.

### 3. Anomoly Detection Strategy
- Isolation Forest:
    - A powerful unsupervised method trained on noise-only data.
    - Leveraged combined features (SVD/NMF components and reconstruction errors).
    - Produced anomaly scores, indicating the likelihood of samples being anomalous.
- Balanced Holdout Set:
    - Created a balanced evaluation dataset combining unseen noise and signals for unbiased model assessment.
- Performance Evaluation:
    - Evaluated using ROC AUC metrics, visualizations of anomaly score distributions, and ROC curves.
    - Initial Results:
        - Baseline approach (basic features) achieved ROC AUC ~0.77.
        - Enhanced approach (SVD/NMF reconstruction features) significantly improved ROC AUC to 0.894, demonstrating substantial performance gains.

### 4. Key Outcomes
- Developed a robust pipeline integrating statistical preprocessing, matrix factorization, dimensionality reduction, and Isolation Forest-based anomaly detection.
- Demonstrated clear performance benefits of incorporating reconstruction errors (SVD/NMF) and advanced feature engineering.
- Provided visualizations, metrics (ROC AUC), and interpretable plots to clarify and validate model performance.
- Outlined a practical, scalable strategy to generalize the approach to handle multiple signal types.

### 5. Future Project Improvements
Given the excellent anomaly detection performance observed, the project proposed extending the approach:
- One-Class per Signal Approach:
    - Training independent Isolation Forest + SVD/NMF models for each signal type separately.
    - New, unlabeled spectrograms could be evaluated against each specialized detector.
    - Practical advantages:
        - Enhanced accuracy via specialized models.
        - Simplified debugging and interpretability.
        - Modular framework allowing easy addition of new signal types.

## Project Structure
```
.
├── data/                             # Signal data directory (not in repo)
├── notebooks/         
│   └── seti_analysis.ipynb           # Main analysis notebook
├── src/
│   ├── anomaly_detection.py          # Anomaly detection utilities
│   ├── cluster_analysis.py           # Cluster analysis utilities
│   ├── dimensionality_reduction.py   # Dimensionality reduction utilities
│   ├── matrix_factorization.py       # Matrix factorization utilities
│   ├── preprocess.py                 # Data preprocessing pipeline
└── requirements.txt                  # Project dependencies
```

## Recommended Interaction
The notebook `seti_analysis.ipynb` is the recommended way to interact with the project. It provides a comprehensive overview of the project, including data preprocessing, dimensionality reduction, matrix factorization, and anomaly detection. The notebook is saved with the outputs so it can be easily reviewed.

If you'd like to run the code yourself, follow the installation instructions below.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/lmarte17/Final-Project-SETI-UL.git
cd Final-Project-SETI-UL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Download the SETI signal dataset from [Kaggle](https://www.kaggle.com/datasets/tentotheminus9/seti-data) and place it in the `data/` directory

2. Go to 'notebooks/seti_analysis.ipynb' and run the notebook.
