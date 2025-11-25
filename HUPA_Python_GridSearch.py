# -*- coding: utf-8 -*-
"""
HUPA_Python_GridSearch.py

Purpose:
    Performs a Grid Search with Cross-Validation (CV) to optimize binary classification models
    (Normal vs. Pathological) using acoustic features.
    
    The workflow matches the MATLAB implementation:
    1. Loads features from CSV.
    2. Splits data into Train (80%) and Hold-out Test (20%).
    3. Defines feature groups (Noise, Perturbation, Tremor, Complexity).
    4. For each group, runs a GridSearchCV on 5 classifiers:
       - Logistic Regression
       - SVM (RBF)
       - Random Forest
       - k-NN
       - MLP (Neural Network)
    5. Evaluates the best models on the Test set and plots ROC curves.

Usage:
    Ensure the input CSV is located at: ./data/HUPA_voice_features_PRN_CPP.csv
    Run via terminal: python HUPA_Python_GridSearch.py

Requirements:
    pandas, numpy, matplotlib, scikit-learn
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Suppress convergence warnings to keep the console clean during GridSearch
warnings.filterwarnings("ignore")

def main():
    # =========================================================================
    # 1. SETUP PATHS & LOAD DATA
    # =========================================================================
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input path relative to the script location
    input_csv_path = os.path.join(script_dir, "data", "HUPA_voice_features_PRN_CPP.csv")
    
    # Check if file exists
    if not os.path.exists(input_csv_path):
        print(f"ERROR: Input file not found at: {input_csv_path}")
        print("Please ensure the CSV file is inside a 'data' folder next to this script.")
        sys.exit(1)

    print(f"Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Validate target column
    if "Label" not in df.columns:
        raise ValueError("Column 'Label' not found in CSV. It must exist and contain binary labels (0/1).")

    # Ensure labels are integers
    y = df["Label"].astype(int)

    # =========================================================================
    # 2. DEFINE FEATURE GROUPS
    # =========================================================================
    
    # Helper to only select columns that actually exist in the CSV
    def existing(cols):
        return [c for c in cols if c in df.columns]

    # --- Group 1: Noise Parameters ---
    noise_features = existing([
        "HNR_mean",  "HNR_std",   # Harmonics-to-Noise Ratio
        "CHNR_mean", "CHNR_std",  # Cepstral HNR
        "GNE_mean",  "GNE_std",   # Glottal to Noise Excitation Ratio
        "NNE_mean",  "NNE_std",   # Normalized Noise Energy
    ])

    # --- Group 2: Perturbation Measures ---
    perturbation_features = existing([
        "CPP",        # Cepstral Peak Prominence (Covarep)
        "rShdB",      # Shimmer (dB)
        "rShim",      # Relative Shimmer (%)
        "rAPQ",       # Amplitude Perturbation Quotient
        "rSAPQ",      # Smoothed APQ
        "rJitta",     # Absolute Jitter
        "rJitt",      # Relative Jitter (%)
        "rRrRAP",     # Relative Average Perturbation
        "rPPQ",       # Pitch Period Perturbation Quotient
        "rSPPQ",      # Smoothed PPQ
    ])

    # --- Group 3: Tremor Parameters ---
    tremor_features = existing([
        "rFTRI",      # Frequency Tremor Intensity Index
        "rATRI",      # Amplitude Tremor Intensity Index
        "rFftr",      # Frequency Tremor Frequency
        "rFatr",      # Amplitude Tremor Frequency
    ])

    # --- Group 4: Complexity / Nonlinear Features ---
    complexity_features = existing([
        # Entropy measures
        "rApEn_mean",    "rApEn_std",
        "rSampEn_mean",  "rSampEn_std",
        "rFuzzyEn_mean", "rFuzzyEn_std",
        "rGSampEn_mean", "rGSampEn_std",
        "rmSampEn_mean", "rmSampEn_std",

        # Fractal / Chaos dimensions
        "CorrDim_mean",  "CorrDim_std",
        "LLE_mean",      "LLE_std",       # Largest Lyapunov Exponent
        "Hurst_mean",    "Hurst_std",     # Hurst Exponent
        "mDFA_mean",     "mDFA_std",      # Detrended Fluctuation Analysis

        # Recurrence & Others
        "RPDE_mean",     "RPDE_std",      # Recurrence Period Density Entropy
        "PE_mean",       "PE_std",        # Permutation Entropy
        "MarkEnt_mean",  "MarkEnt_std",   # Markov Entropy
    ])

    # Store groups in a dictionary for iteration
    feature_groups = {
        "Noise":        noise_features,
        "Perturbation": perturbation_features,
        "Tremor":       tremor_features,
        "Complexity":   complexity_features,
    }

    # =========================================================================
    # 3. SPLIT DATA (TRAIN / TEST)
    # =========================================================================
    # We split by index to ensure all feature groups use exactly the same samples
    indices = np.arange(len(df))
    
    # 80% Train, 20% Test, Stratified to maintain class balance
    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"\nData Split: {len(idx_train)} Training samples, {len(idx_test)} Test samples.")

    # =========================================================================
    # 4. DEFINE MODEL PIPELINES & HYPERPARAMETER GRIDS
    # =========================================================================
    # Each pipeline includes:
    # 1. Imputer: Handles missing values (NaNs) by filling with median.
    # 2. Scaler: Standardizes features (Z-score), essential for SVM, KNN, MLP, LR.
    # 3. Classifier: The model itself.
    
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipelines_and_grids = {
        # --- Logistic Regression ---
        "logreg": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=42)),
            ]),
            {
                "clf__penalty": ["l2", "l1"],
                "clf__C": [0.01, 0.1, 1, 10, 30],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        
        # --- SVM (RBF Kernel) ---
        "svc_rbf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, max_iter=50000, random_state=42)),
            ]),
            {
                "clf__C":      [0.1, 1, 10, 30, 100],
                "clf__gamma":  ["scale", 0.001, 0.01, 0.1],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        
        # --- Random Forest ---
        # Note: RF doesn't strictly need scaling, but Imputation is required.
        "rf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        
        # --- k-Nearest Neighbors ---
        "knn": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [3, 5, 7, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),
        
        # --- MLP (Neural Network) ---
        "mlp": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)),
            ]),
            {
                "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "clf__alpha": [1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 1e-2],
                "clf__activation": ["relu", "tanh"],
            },
        ),
    }

    # Display names for plots
    pretty_names = {
        "logreg":  "Logistic Regression",
        "svc_rbf": "SVM (RBF)",
        "knn":     "k-NN",
        "rf":      "Random Forest",
        "mlp":     "Neural Network",
    }
    
    # Consistent order for plotting
    model_order = ["logreg", "svc_rbf", "rf", "knn", "mlp"]

    # =========================================================================
    # 5. MAIN LOOP: GRID SEARCH & EVALUATION
    # =========================================================================
    results_by_group = {}
    roc_data_by_group = {}

    for group_name, cols in feature_groups.items():
        if len(cols) == 0:
            print(f"\n[SKIP] Group '{group_name}' has no valid columns.")
            continue

        print(f"\n==================================================")
        print(f" PROCESSING GROUP: {group_name} ({len(cols)} features)")
        print(f"==================================================")
        
        # Subset data for this specific group
        X_group = df.loc[:, cols]

        X_train = X_group.iloc[idx_train]
        X_test  = X_group.iloc[idx_test]
        y_train = y.iloc[idx_train]
        y_test  = y.iloc[idx_test]

        group_summary = []
        group_roc_curves = {}

        for model_key in model_order:
            if model_key not in pipelines_and_grids: continue
            
            pipe, grid = pipelines_and_grids[model_key]
            
            print(f" -> Tuning {pretty_names[model_key]}...")

            # Run Grid Search with Cross Validation
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="roc_auc",
                n_jobs=-1,      # Use all CPU cores
                cv=cv_scheme,
                refit=True,     # Refit best model on full train set
                verbose=0
            )
            
            gs.fit(X_train, y_train)
            
            # --- Evaluation on HOLD-OUT TEST SET ---
            best_model = gs.best_estimator_
            
            # Get probabilities for AUC calculation
            # Try predict_proba first, fallback to decision_function for SVM/Logistic if needed
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]
            else:
                # Fallback for models that might not have probability enabled
                scores = best_model.decision_function(X_test)
                # Min-Max scale scores to 0-1 range for ROC function compatibility
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

            # Calculate Metrics
            test_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # Store results
            group_summary.append({
                "Group": group_name,
                "Model": pretty_names[model_key],
                "CV_AUC_Mean": gs.best_score_,
                "Test_AUC": test_auc,
                "Best_Params": str(gs.best_params_)
            })
            
            group_roc_curves[model_key] = (fpr, tpr, test_auc)
            
            print(f"    Best CV AUC: {gs.best_score_:.3f} | Test AUC: {test_auc:.3f}")

        # Convert results to DataFrame and sort by Test AUC
        results_df = pd.DataFrame(group_summary)
        results_df = results_df.sort_values(by="Test_AUC", ascending=False)
        results_by_group[group_name] = results_df
        roc_data_by_group[group_name] = group_roc_curves

    # =========================================================================
    # 6. VISUALIZATION (ROC CURVES)
    # =========================================================================
    # Create a 2x2 subplot layout for the 4 specific groups
    groups_to_plot = ["Noise", "Perturbation", "Tremor", "Complexity"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel() # Flatten to 1D array for easy iteration

    for i, gname in enumerate(groups_to_plot):
        ax = axes[i]
        
        if gname not in roc_data_by_group:
            ax.set_visible(False) # Hide plot if group data is missing
            continue

        roc_curves = roc_data_by_group[gname]
        
        # Plot each model's curve
        for model_key in model_order:
            if model_key in roc_curves:
                fpr, tpr, auc = roc_curves[model_key]
                label_text = f"{pretty_names[model_key]} (AUC={auc:.2f})"
                ax.plot(fpr, tpr, lw=2, label=label_text)

        # Plot diagonal random guess line
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{gname} Features")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 7. SAVE RESULTS TO CSV
    # =========================================================================
    # Combine all group results into one large DataFrame
    if results_by_group:
        final_summary = pd.concat(results_by_group.values(), ignore_index=True)
        
        # Organize columns cleanly
        cols_order = ["Group", "Model", "Test_AUC", "CV_AUC_Mean", "Best_Params"]
        final_summary = final_summary[cols_order]

        # Define output path
        output_csv_path = os.path.join(script_dir, "data", "HUPA_Python_Results_Summary.csv")
        
        final_summary.to_csv(output_csv_path, index=False)
        print(f"\n[DONE] Summary results saved to: {output_csv_path}")
        
        # Print top model per group to console
        print("\n=== TOP MODEL PER GROUP ===")
        for gname, rdf in results_by_group.items():
            best_row = rdf.iloc[0]
            print(f"{gname}: {best_row['Model']} (AUC={best_row['Test_AUC']:.3f})")

if __name__ == "__main__":
    main()