import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import mlflow
import mlflow.sklearn
from pipeline import KeystrokeAuthPipeline

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_thresh = thresholds[idx]
    return eer, eer_thresh, fpr, tpr

def train(args):
    data_path = 'data/DSL-StrongPasswordData.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    metadata_cols = ['subject', 'sessionIndex', 'rep']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    print(f"Features: {len(feature_cols)} timing attributes")

    # Target definition (s002 = genuine 1, all others = impostor 0)
    enrolled_subject = args.subject
    df['target'] = (df['subject'] == enrolled_subject).astype(int)

    # Session-based partition
    # Train: Sessions 1, 2, 3
    # Val:   Session 4 (for hyperparameter evaluation and EER threshold tuning)
    # Test:  Sessions 5, 6, 7, 8 (unseen holdout)
    train_mask = df['sessionIndex'].isin([1, 2, 3])
    val_mask   = df['sessionIndex'] == 4
    test_mask  = df['sessionIndex'].isin([5, 6, 7, 8])

    X_train_df = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'target'].values

    X_val_df = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, 'target'].values

    X_test_df = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'target'].values

    print(f"Train samples (S1-3): {len(y_train)} (Genuine: {y_train.sum()})")
    print(f"Val samples   (S4):   {len(y_val)} (Genuine: {y_val.sum()})")
    print(f"Test samples  (S5-8): {len(y_test)} (Genuine: {y_test.sum()})")

    # Fit Scaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled   = scaler.transform(X_val_df)
    X_test_scaled  = scaler.transform(X_test_df)

    # Optional PCA
    pca = None
    if args.use_pca:
        pca = PCA(n_components=args.pca_components, random_state=42)
        X_train_proc = pca.fit_transform(X_train_scaled)
        X_val_proc   = pca.transform(X_val_scaled)
        X_test_proc  = pca.transform(X_test_scaled)
        print(f"PCA retained {X_train_proc.shape[1]} components")
    else:
        X_train_proc = X_train_scaled
        X_val_proc   = X_val_scaled
        X_test_proc  = X_test_scaled

    # Algorithm instantiation
    if args.model_type == 'logistic_regression':
        # Baseline unweighted (reproduces midterm 99% FRR failure mode)
        model = LogisticRegression(C=args.C, max_iter=1000, random_state=42)
    elif args.model_type == 'logistic_regression_balanced':
        model = LogisticRegression(C=args.C, class_weight='balanced', max_iter=1000, random_state=42)
    elif args.model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif args.model_type == 'svm_rbf':
        model = SVC(
            C=args.C,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Set MLflow tracking to SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Keystroke-Authentication")

    run_name = args.run_name or f"{args.model_type}_{'pca' if args.use_pca else 'raw'}"
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Training {args.model_type} ---")
        model.fit(X_train_proc, y_train)

        # Probabilities on validation set
        if hasattr(model, 'predict_proba'):
            val_scores = model.predict_proba(X_val_proc)[:, 1]
            test_scores = model.predict_proba(X_test_proc)[:, 1]
        else:
            val_scores = model.decision_function(X_val_proc)
            test_scores = model.decision_function(X_test_proc)

        # Calculate EER and optimal threshold on Validation set (Session 4)
        val_eer, val_eer_thresh, fpr_val, tpr_val = calculate_eer(y_val, val_scores)
        print(f"Validation EER: {val_eer*100:.2f}% at threshold: {val_eer_thresh:.4f}")

        # Choose operating threshold
        if args.model_type == 'logistic_regression' and not args.tune_threshold:
            # Midterm uncalibrated default
            operational_threshold = 0.5
        elif args.tune_threshold:
            operational_threshold = float(val_eer_thresh)
        else:
            operational_threshold = 0.5

        # Evaluate on Test Set (Sessions 5-8)
        y_test_pred = (test_scores >= operational_threshold).astype(int)

        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, test_scores)
        test_eer, test_eer_thresh, fpr_test, tpr_test = calculate_eer(y_test, test_scores)

        print("\n=== TEST SET EVALUATION (Sessions 5–8) ===")
        print(f"Operational Threshold : {operational_threshold:.4f}")
        print(f"Confusion Matrix      : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy              : {acc*100:.2f}%")
        print(f"Precision             : {prec:.4f}")
        print(f"Recall                : {rec:.4f}")
        print(f"F1-Score              : {f1:.4f}")
        print(f"FAR (False Accept)    : {far*100:.2f}%")
        print(f"FRR (False Reject)    : {frr*100:.2f}%")
        print(f"ROC-AUC               : {auc:.4f}")
        print(f"Test EER              : {test_eer*100:.2f}%")

        # Log parameters to MLflow
        mlflow.log_params({
            "model_type": args.model_type,
            "subject": args.subject,
            "use_pca": args.use_pca,
            "pca_components": args.pca_components if args.use_pca else "None",
            "C": args.C if 'logistic' in args.model_type or 'svm' in args.model_type else "N/A",
            "n_estimators": args.n_estimators if 'forest' in args.model_type else "N/A",
            "max_depth": args.max_depth if 'forest' in args.model_type else "N/A",
            "operational_threshold": operational_threshold,
            "train_sessions": "1,2,3",
            "val_sessions": "4",
            "test_sessions": "5,6,7,8"
        })

        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "far": far,
            "frr": frr,
            "roc_auc": auc,
            "eer": test_eer,
            "val_eer": val_eer,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

        # Plot ROC curve artifact
        os.makedirs("model", exist_ok=True)
        roc_plot_path = f"model/roc_{args.model_type}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_test, tpr_test, label=f"{args.model_type} (AUC = {auc:.3f})", color='steelblue', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        ax.set_xlabel("False Positive Rate (FAR)")
        ax.set_ylabel("True Positive Rate (1 - FRR)")
        ax.set_title(f"ROC Curve — {args.model_type}")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(roc_plot_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(roc_plot_path)

        # Log sklearn model using cloudpickle serialization format
        try:
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle"
            )
        except TypeError:
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    serialization_format="cloudpickle"
                )
            except Exception as e:
                print(f"Notice on model logging: {e}")

        # Serialize production pipeline if requested
        if args.save_model:
            if args.refit_full:
                print("\nRefitting scaler and model on Sessions 1–4 for serialization...")
                full_train_mask = df['sessionIndex'].isin([1, 2, 3, 4])
                X_full_train = df.loc[full_train_mask, feature_cols]
                y_full_train = df.loc[full_train_mask, 'target'].values

                prod_scaler = StandardScaler()
                X_full_scaled = prod_scaler.fit_transform(X_full_train)

                prod_pca = None
                if args.use_pca:
                    prod_pca = PCA(n_components=args.pca_components, random_state=42)
                    X_full_proc = prod_pca.fit_transform(X_full_scaled)
                else:
                    X_full_proc = X_full_scaled

                model.fit(X_full_proc, y_full_train)
                save_scaler = prod_scaler
                save_pca = prod_pca
                train_sessions_desc = "Sessions 1–4"
            else:
                print("\nSaving strictly paired production pipeline (fitted on Sessions 1–3, calibrated on Session 4)...")
                save_scaler = scaler
                save_pca = pca
                train_sessions_desc = "Sessions 1–3"

            pipeline = KeystrokeAuthPipeline(
                scaler=save_scaler,
                pca=save_pca,
                classifier=model,
                feature_names=feature_cols,
                threshold=operational_threshold,
                metadata={
                    "model_type": args.model_type,
                    "subject": args.subject,
                    "training_sessions": train_sessions_desc,
                    "calibration_session": "Session 4",
                    "accuracy": acc,
                    "f1_score": f1,
                    "far": far,
                    "frr": frr,
                    "roc_auc": auc,
                    "eer": test_eer
                }
            )

            joblib_path = "model/model.joblib"
            joblib.dump(pipeline, joblib_path)
            print(f"Successfully serialized production pipeline to {joblib_path}")
            mlflow.log_artifact(joblib_path)

    return acc, f1, far, frr, auc, test_eer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Keystroke Dynamics Authentication Models")
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['logistic_regression', 'logistic_regression_balanced', 'random_forest', 'svm_rbf'],
                        help="Algorithm choice")
    parser.add_argument('--subject', type=str, default='s002', help="Enrolled subject ID")
    parser.add_argument('--n_estimators', type=int, default=100, help="Random Forest estimators")
    parser.add_argument('--max_depth', type=int, default=6, help="Random Forest max depth")
    parser.add_argument('--C', type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument('--use_pca', action='store_true', help="Apply PCA dimensionality reduction")
    parser.add_argument('--pca_components', type=float, default=0.95, help="PCA variance ratio or n_components")
    parser.add_argument('--tune_threshold', action='store_true', help="Tune threshold at EER on validation set")
    parser.add_argument('--save_model', action='store_true', help="Save pipeline to model/model.joblib")
    parser.add_argument('--refit_full', action='store_true', help="Refit on Sessions 1-4 before saving (default False: strictly pairs Sessions 1-3 model with Session 4 threshold)")
    parser.add_argument('--run_name', type=str, default=None, help="MLflow run name")

    args = parser.parse_args()
    train(args)
