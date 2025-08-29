import numpy as np
import pandas as pd
import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr

# Load the dataset
dataset = openml.datasets.get_dataset(1494)  # QSAR_biodegradation dataset
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

# Define the original attribute names in the correct order
original_attribute_names = [
    "SpMax_L",  # 1) Leading eigenvalue from Laplace matrix
    "J_Dz_e",  # 2) Balaban-like index from Barysz matrix weighted by Sanderson electronegativity
    "nHM",  # 3) Number of heavy atoms
    "F01_N_N",  # 4) Frequency of N-N at topological distance 1
    "F04_C_N",  # 5) Frequency of C-N at topological distance 4
    "NssssC",  # 6) Number of atoms of type ssssC
    "nCb_",  # 7) Number of substituted benzene C(sp2)
    "C_percent",  # 8) Percentage of C atoms
    "nCp",  # 9) Number of terminal primary C(sp3)
    "nO",  # 10) Number of oxygen atoms
    "F03_C_N",  # 11) Frequency of C-N at topological distance 3
    "SdssC",  # 12) Sum of dssC E-states
    "HyWi_B_m",  # 13) Hyper-Wiener-like index (log function) from Burden matrix weighted by mass
    "LOC",  # 14) Lopping centric index
    "SM6_L",  # 15) Spectral moment of order 6 from Laplace matrix
    "F03_C_O",  # 16) Frequency of C - O at topological distance 3
    "Me",  # 17) Mean atomic Sanderson electronegativity (scaled on Carbon atom)
    "Mi",  # 18) Mean first ionization potential (scaled on Carbon atom)
    "nN_N",  # 19) Number of N hydrazines
    "nArNO2",  # 20) Number of nitro groups (aromatic)
    "nCRX3",  # 21) Number of CRX3
    "SpPosA_B_p",  # 22) Normalized spectral positive sum from Burden matrix weighted by polarizability
    "nCIR",  # 23) Number of circuits
    "B01_C_Br",  # 24) Presence/absence of C - Br at topological distance 1
    "B03_C_Cl",  # 25) Presence/absence of C - Cl at topological distance 3
    "N_073",  # 26) Ar2NH / Ar3N / Ar2N-Al / R..N..R
    "SpMax_A",  # 27) Leading eigenvalue from adjacency matrix (Lovasz-Pelikan index)
    "Psi_i_1d",  # 28) Intrinsic state pseudoconnectivity index - type 1d
    "B04_C_Br",  # 29) Presence/absence of C - Br at topological distance 4
    "SdO",  # 30) Sum of dO E-states
    "TI2_L",  # 31) Second Mohar index from Laplace matrix
    "nCrt",  # 32) Number of ring tertiary C(sp3)
    "C_026",  # 33) R--CX--R
    "F02_C_N",  # 34) Frequency of C - N at topological distance 2
    "nHDon",  # 35) Number of donor atoms for H-bonds (N and O)
    "SpMax_B_m",  # 36) Leading eigenvalue from Burden matrix weighted by mass
    "Psi_i_A",  # 37) Intrinsic state pseudoconnectivity index - type S average
    "nN",  # 38) Number of Nitrogen atoms
    "SM6_B_m",  # 39) Spectral moment of order 6 from Burden matrix weighted by mass
    "nArCOOR",  # 40) Number of esters (aromatic)
    "nX"  # 41) Number of halogen atoms
]

# Use original attribute names for the DataFrame
X = pd.DataFrame(X, columns=original_attribute_names)
y = pd.Series(y)

# Function to format feature importance
def format_features_with_importance(features, importances):
    result = []
    for feature in features:
        importance = importances[feature]
        result.append(f"{feature} ({importance:.4f})")
    return ", ".join(result)

# Function to perform feature selection with random forest
def rf_feature_selection(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Function to perform feature selection with xgboost
def xgb_feature_selection(X, y):
    model = XGBClassifier(random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Function to perform feature selection with logistic regression
def lr_feature_selection(X, y):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False)

# Feature Agglomeration - considering both correlation and variance
def feature_agglomeration_selection(X):
    # Create 10 clusters to identify related features
    n_clusters = 10
    
    # Use feature agglomeration to cluster features
    agglomerator = FeatureAgglomeration(n_clusters=n_clusters)
    X_reduced = agglomerator.fit_transform(X)
    
    # Calculate variance for each feature
    variances = X.var()
    
    # Initialize feature_scores dictionary
    feature_scores = {}
    
    # Calculate correlation of each feature with its cluster center
    for cluster_idx in range(n_clusters):
        # Find all features in this cluster
        cluster_features = [feature for feature, label in 
                          zip(X.columns, agglomerator.labels_) if label == cluster_idx]
        
        if cluster_features:
            # Get the original data for these features
            cluster_data = X[cluster_features]
            
            # Get the transformed data for this cluster
            transformed_value = X_reduced[:, cluster_idx].reshape(-1, 1)
            
            # Calculate correlation for each feature with the cluster center
            for feature in cluster_features:
                corr = abs(np.corrcoef(cluster_data[feature], transformed_value.flatten())[0, 1])
                feature_scores[feature] = corr
    
    # Convert to Series
    corr_scores = pd.Series(feature_scores)
    
    # Normalize both scores to 0-1 range
    normalized_corr = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min())
    normalized_var = (variances - variances.min()) / (variances.max() - variances.min())
    
    # Combine correlation scores and variances with equal weight
    combined_scores = 0.5 * normalized_corr + 0.5 * normalized_var
    
    # Return the combined scores sorted
    return combined_scores.sort_values(ascending=False)

# Highly Variable Gene Selection - unsupervised
def highly_variable_selection(X):
    # Calculate variance for each feature
    variances = X.var()
    
    # Sort features by variance and return
    return variances.sort_values(ascending=False)

# Spearman Correlation - supervised but not using RF
def spearman_selection(X, y):
    # Calculate Spearman correlation for each feature with the target
    correlations = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations[col] = abs(corr)  # Use absolute correlation
    
    # Convert to Series, sort by absolute correlation, and return
    return pd.Series(correlations).sort_values(ascending=False)

# Function to evaluate model with cross-validation
def evaluate_model(X, y, model, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), scoring='accuracy')
    return cv_scores.mean()

# Results table
results = {
    'Algorithm': [],
    'Top 10 Features CV Accuracy': [],
    'Top 1 Feature Removed': [],
    'Set 1 (top 9 from top 10)': [],
    'Set 2 (top 9 from reduced)': []
}

# Random Forest
print("Evaluating Random Forest...")
rf_importances = rf_feature_selection(X, y)
top10_rf = rf_importances.index[:10]
top1_rf = rf_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_rf = list(top10_rf[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_rf]
X_remaining = X[remaining_features]
rf_remaining = RandomForestClassifier(random_state=42)
rf_remaining.fit(X_remaining, y)
remaining_importances = pd.Series(rf_remaining.feature_importances_, index=remaining_features).sort_values(ascending=False)
# Set 2: Top 9 from remaining features
set2_rf = list(remaining_importances.index[:9])

# Evaluate using Random Forest
rf_model = RandomForestClassifier(random_state=42)
top10_acc_rf = evaluate_model(X[top10_rf], y, rf_model)

results['Algorithm'].append('Random Forest')
results['Top 10 Features CV Accuracy'].append(top10_acc_rf)
results['Top 1 Feature Removed'].append(f"{top1_rf} ({rf_importances[top1_rf]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_rf, rf_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_rf, remaining_importances))

# XGBoost
print("Evaluating XGBoost...")
xgb_importances = xgb_feature_selection(X, y)
top10_xgb = xgb_importances.index[:10]
top1_xgb = xgb_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_xgb = list(top10_xgb[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_xgb]
X_remaining = X[remaining_features]
xgb_remaining = XGBClassifier(random_state=42)
xgb_remaining.fit(X_remaining, y)
remaining_importances = pd.Series(xgb_remaining.feature_importances_, index=remaining_features).sort_values(ascending=False)
# Set 2: Top 9 from remaining features
set2_xgb = list(remaining_importances.index[:9])

# Evaluate using XGBoost
xgb_model = XGBClassifier(random_state=42)
top10_acc_xgb = evaluate_model(X[top10_xgb], y, xgb_model)

results['Algorithm'].append('XGBoost')
results['Top 10 Features CV Accuracy'].append(top10_acc_xgb)
results['Top 1 Feature Removed'].append(f"{top1_xgb} ({xgb_importances[top1_xgb]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_xgb, xgb_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_xgb, remaining_importances))

# Logistic Regression
print("Evaluating Logistic Regression...")
lr_importances = lr_feature_selection(X, y)
top10_lr = lr_importances.index[:10]
top1_lr = lr_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_lr = list(top10_lr[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_lr]
X_remaining = X[remaining_features]
lr_remaining = LogisticRegression(random_state=42, max_iter=1000)
lr_remaining.fit(X_remaining, y)
remaining_importances = pd.Series(np.abs(lr_remaining.coef_[0]), index=remaining_features).sort_values(ascending=False)
# Set 2: Top 9 from remaining features
set2_lr = list(remaining_importances.index[:9])

# Evaluate using Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
top10_acc_lr = evaluate_model(X[top10_lr], y, lr_model)

results['Algorithm'].append('Logistic Regression')
results['Top 10 Features CV Accuracy'].append(top10_acc_lr)
results['Top 1 Feature Removed'].append(f"{top1_lr} ({lr_importances[top1_lr]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_lr, lr_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_lr, remaining_importances))

# Feature Agglomeration - unsupervised
print("Evaluating Feature Agglomeration...")
fa_importances = feature_agglomeration_selection(X)
top10_fa = fa_importances.index[:10]
top1_fa = fa_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_fa = list(top10_fa[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_fa]
X_remaining = X[remaining_features]
# Apply feature agglomeration on remaining features
fa_remaining_importances = feature_agglomeration_selection(X_remaining)
# Set 2: Top 9 from remaining features
set2_fa = list(fa_remaining_importances.index[:9])

# Evaluate using Random Forest (only for evaluation, not selection)
rf_model = RandomForestClassifier(random_state=42)
top10_acc_fa = evaluate_model(X[top10_fa], y, rf_model)

results['Algorithm'].append('Feature Agglomeration')
results['Top 10 Features CV Accuracy'].append(top10_acc_fa)
results['Top 1 Feature Removed'].append(f"{top1_fa} ({fa_importances[top1_fa]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_fa, fa_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_fa, fa_remaining_importances))

# Highly Variable Gene Selection - unsupervised
print("Evaluating Highly Variable Gene Selection...")
hv_importances = highly_variable_selection(X)
top10_hv = hv_importances.index[:10]
top1_hv = hv_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_hv = list(top10_hv[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_hv]
X_remaining = X[remaining_features]
# Calculate variance for remaining features
hv_remaining_importances = highly_variable_selection(X_remaining)
# Set 2: Top 9 from remaining features
set2_hv = list(hv_remaining_importances.index[:9])

# Evaluate using Random Forest (only for evaluation, not selection)
rf_model = RandomForestClassifier(random_state=42)
top10_acc_hv = evaluate_model(X[top10_hv], y, rf_model)

results['Algorithm'].append('Highly Variable Selection')
results['Top 10 Features CV Accuracy'].append(top10_acc_hv)
results['Top 1 Feature Removed'].append(f"{top1_hv} ({hv_importances[top1_hv]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_hv, hv_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_hv, hv_remaining_importances))

# Spearman Correlation
print("Evaluating Spearman Correlation...")
sp_importances = spearman_selection(X, y)
top10_sp = sp_importances.index[:10]
top1_sp = sp_importances.index[0]
# Set 1: Remove top 1 from top 10 to get top 9
set1_sp = list(top10_sp[1:10])

# Create a reduced dataset without the top 1 feature
remaining_features = [f for f in X.columns if f != top1_sp]
X_remaining = X[remaining_features]
# Calculate Spearman correlation for remaining features
sp_remaining_importances = spearman_selection(X_remaining, y)
# Set 2: Top 9 from remaining features
set2_sp = list(sp_remaining_importances.index[:9])

# Evaluate using Random Forest (only for evaluation, not selection)
rf_model = RandomForestClassifier(random_state=42)
top10_acc_sp = evaluate_model(X[top10_sp], y, rf_model)

results['Algorithm'].append('Spearman Correlation')
results['Top 10 Features CV Accuracy'].append(top10_acc_sp)
results['Top 1 Feature Removed'].append(f"{top1_sp} ({sp_importances[top1_sp]:.4f})")
results['Set 1 (top 9 from top 10)'].append(format_features_with_importance(set1_sp, sp_importances))
results['Set 2 (top 9 from reduced)'].append(format_features_with_importance(set2_sp, sp_remaining_importances))

# Create results DataFrame and display
results_df = pd.DataFrame(results)
# Adjust display options to show full content of columns
pd.set_option('display.max_colwidth', None)
print("\nResults Summary:")
print(results_df[['Algorithm', 'Top 10 Features CV Accuracy', 'Top 1 Feature Removed']])

print("\nFeature Sets Comparison:")
for i, row in results_df.iterrows():
    print(f"\n{row['Algorithm']}:")
    print(f"  Top 1 Feature Removed: {row['Top 1 Feature Removed']}")
    print(f"  Set 1 (top 9 from top 10): {row['Set 1 (top 9 from top 10)']}")
    print(f"  Set 2 (top 9 from reduced): {row['Set 2 (top 9 from reduced)']}")
