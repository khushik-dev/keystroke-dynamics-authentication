import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def run_drift_check(data_path="data/DSL-StrongPasswordData.csv",
                    log_path="monitoring/prediction_logs.jsonl",
                    output_report="monitoring/drift_report.json"):
    """
    Compares production inference input distributions against the baseline
    training distribution (Sessions 1–4) across all 31 timing features.
    Computes Mean, Std, % Shift, and Kolmogorov-Smirnov (KS) test statistics.
    """
    print("=" * 65)
    print("KEYSTROKE DYNAMICS DATA DRIFT & COVARIATE SHIFT AUDIT")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Reference dataset not found at {data_path}")

    df_ref = pd.read_csv(data_path)
    timing_features = [c for c in df_ref.columns if c not in ['subject', 'sessionIndex', 'rep']]

    # Reference distribution: Training sessions 1-4 for s002
    train_ref = df_ref[df_ref['sessionIndex'].isin([1, 2, 3, 4])][timing_features]

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        print(f"No prediction logs found at {log_path}. Creating sample log entries for verification...")
        # Simulate test session samples if no live logs exist yet
        test_samples = df_ref[df_ref['sessionIndex'].isin([5, 6])][timing_features].head(50)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for _, row in test_samples.iterrows():
                entry = {
                    "timestamp": "2026-11-17T00:00:00",
                    "prediction": 0,
                    "match_score": 0.05,
                    "features": row.to_dict()
                }
                f.write(json.dumps(entry) + "\n")

    # Read production logs
    logged_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if "features" in entry:
                    logged_records.append(entry["features"])

    if not logged_records:
        print("No valid feature records found in log file.")
        return

    df_prod = pd.DataFrame(logged_records)[timing_features]
    print(f"Reference Baseline Samples (Train S1-4): {len(train_ref)}")
    print(f"Production Inference Samples (Logged)   : {len(df_prod)}")

    results = []
    drift_detected_count = 0

    for col in timing_features:
        ref_vals = train_ref[col].values
        prod_vals = df_prod[col].dropna().values

        ref_mean, ref_std = float(np.mean(ref_vals)), float(np.std(ref_vals))
        prod_mean, prod_std = float(np.mean(prod_vals)), float(np.std(prod_vals))

        mean_diff = prod_mean - ref_mean
        pct_shift = (abs(mean_diff) / (abs(ref_mean) + 1e-6)) * 100

        # Kolmogorov-Smirnov 2-sample test
        ks_stat, p_value = stats.ks_2samp(ref_vals, prod_vals)
        is_drift = bool(p_value < 0.05 and pct_shift > 15.0)
        if is_drift:
            drift_detected_count += 1

        results.append({
            "feature": col,
            "ref_mean": round(ref_mean, 4),
            "ref_std": round(ref_std, 4),
            "prod_mean": round(prod_mean, 4),
            "prod_std": round(prod_std, 4),
            "pct_shift": round(pct_shift, 2),
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": is_drift
        })

    report = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "reference_samples": len(train_ref),
        "production_samples": len(df_prod),
        "total_features": len(timing_features),
        "features_with_drift": drift_detected_count,
        "overall_status": "DRIFT_DETECTED" if drift_detected_count > 5 else "STABLE",
        "feature_metrics": results
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Drift Audit Summary ---")
    print(f"Overall System Status   : {report['overall_status']}")
    print(f"Features Showing Drift  : {drift_detected_count} / {len(timing_features)}")
    print(f"Detailed JSON report saved to: {output_report}")

    # Display top 5 shifted features
    df_res = pd.DataFrame(results).sort_values(by="pct_shift", ascending=False)
    print("\nTop 5 Features by Distribution Shift:")
    print(df_res[['feature', 'ref_mean', 'prod_mean', 'pct_shift', 'p_value', 'drift_detected']].head(5).to_string(index=False))

    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Keystroke Dynamics Data Drift Audit")
    parser.add_argument("--data_path", type=str, default="data/DSL-StrongPasswordData.csv")
    parser.add_argument("--log_path", type=str, default="monitoring/prediction_logs.jsonl")
    parser.add_argument("--output_report", type=str, default="monitoring/drift_report.json")
    args = parser.parse_args()

    run_drift_check(args.data_path, args.log_path, args.output_report)
