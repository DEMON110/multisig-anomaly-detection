"""
Step 06: Evaluation and Statistical Inference
===============================================
Compute all metrics with bootstrap confidence intervals and 
Bonferroni-corrected significance tests.

Metrics (exact from paper):
  - ROC-AUC, PR-AUC, F1, Accuracy, Precision, Recall, MCC

Statistical rigor:
  - Bootstrap: B = 1,000, percentile method
  - Paired t-tests: Bonferroni correction (α_adj = 0.05/7 ≈ 0.007)
  - Wilcoxon signed-rank: non-parametric robustness check
"""

import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix
)
from scipy import stats

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Exact performance values from paper for verification
PAPER_METRICS = {
    "lightgbm": {
        "roc_auc": 0.9971,
        "pr_auc": 0.9865,
        "f1": 0.9739,
        "accuracy": 0.9949,
        "precision": 0.9927,
        "recall": 0.9513,
        "mcc": 0.9690,
    },
    "xgboost": {
        "roc_auc": 0.9976,
        "pr_auc": 0.9876,
        "f1": 0.9707,
        "accuracy": 0.9951,
        "precision": 0.9954,
        "recall": 0.9509,
        "mcc": 0.9703,
    },
    "random_forest": {
        "roc_auc": 0.9959,
        "pr_auc": 0.9853,
        "f1": 0.9739,
        "accuracy": 0.9953,
        "precision": 0.9991,
        "recall": 0.9495,
        "mcc": 0.9715,
    },
    "graphsage": {
        "roc_auc": 0.9824,
        "pr_auc": 0.9663,
        "f1": 0.9045,
        "accuracy": 0.9817,
        "precision": 0.8617,
        "recall": 0.9517,
        "mcc": 0.8957,
    },
    "mlp": {
        "roc_auc": 0.9657,
        "pr_auc": 0.9174,
        "f1": 0.8687,
        "accuracy": 0.9761,
        "precision": 0.8708,
        "recall": 0.8666,
        "mcc": 0.8556,
    },
    "gat": {
        "roc_auc": 0.9415,
        "pr_auc": 0.8665,
        "f1": 0.6717,
        "accuracy": 0.9240,
        "precision": 0.5540,
        "recall": 0.8530,
        "mcc": 0.6498,
    },
    "logistic_regression": {
        "roc_auc": 0.9290,
        "pr_auc": 0.8530,
        "f1": 0.6652,
        "accuracy": 0.9214,
        "precision": 0.5430,
        "recall": 0.8675,
        "mcc": 0.6481,
    },
    "isolation_forest": {
        "roc_auc": 0.9020,
        "pr_auc": 0.6256,
        "f1": 0.5876,
        "accuracy": 0.8904,
        "precision": 0.4434,
        "recall": 0.7946,
        "mcc": 0.5408,
    },
}


def compute_all_metrics(y_true, y_pred_proba, y_pred, threshold=0.5):
    """Compute all 7 metrics used in the paper."""
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_metrics(y_true, y_pred_proba, y_pred, 
                      n_iterations=1000, confidence_level=0.95,
                      seed=456):
    """
    Non-parametric bootstrap confidence intervals.
    
    Paper: B = 1,000, percentile method for 95% CIs.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    bootstrap_results = {metric: [] for metric in 
                         ["roc_auc", "pr_auc", "f1", "accuracy", 
                          "precision", "recall", "mcc"]}
    
    for b in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y_true[indices]
        proba_b = y_pred_proba[indices]
        pred_b = y_pred[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue
        
        metrics_b = compute_all_metrics(y_b, proba_b, pred_b)
        for metric, value in metrics_b.items():
            bootstrap_results[metric].append(value)
    
    # Compute percentile CIs
    results = {}
    alpha = 1 - confidence_level
    for metric, values in bootstrap_results.items():
        values_arr = np.array(values)
        results[f"{metric}_mean"] = np.mean(values_arr)
        results[f"{metric}_std"] = np.std(values_arr)
        results[f"{metric}_ci_lower"] = np.percentile(values_arr, alpha/2 * 100)
        results[f"{metric}_ci_upper"] = np.percentile(values_arr, (1 - alpha/2) * 100)
    
    return results


def paired_significance_test(y_true, proba_lightgbm, proba_baseline, baseline_name):
    """
    Paired one-sided t-test with Bonferroni correction.
    
    Paper:
      H_0: μ_LightGBM = μ_baseline
      H_1: μ_LightGBM > μ_baseline
      α_adj = 0.05 / 7 ≈ 0.007
    """
    # Per-sample predicted probabilities
    diff = proba_lightgbm - proba_baseline
    
    # Paired t-test (one-sided)
    t_stat, p_value_two_sided = stats.ttest_rel(proba_lightgbm, proba_baseline)
    
    # Convert to one-sided
    if np.mean(diff) > 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)
    
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(proba_lightgbm, proba_baseline, 
                                          alternative="greater")
    except ValueError:
        w_stat, w_pvalue = np.nan, np.nan
    
    # Bonferroni threshold
    alpha_adj = 0.05 / 7  # ≈ 0.007
    significant = p_value_one_sided < alpha_adj
    
    return {
        "baseline": baseline_name,
        "delta_auc": np.mean(diff),
        "t_statistic": t_stat,
        "p_value_two_sided": p_value_two_sided,
        "p_value_one_sided": p_value_one_sided,
        "wilcoxon_p": w_pvalue,
        "bonferroni_threshold": alpha_adj,
        "significant": significant,
    }


def main():
    """Run full evaluation pipeline."""
    test = pd.read_csv("data/processed/test_features.csv")
    features = CONFIG["features"]["raw_onchain_metrics"] + \
               CONFIG["features"]["temporal_indicators"] + \
               CONFIG["features"]["normalized_deviation"]
    
    X_test = test[features].values
    y_test = test["is_anomaly"].values
    
    # Load models
    model_names = ["lightgbm", "xgboost", "random_forest", 
                   "graphsage", "mlp", "gat", "logistic_regression", 
                   "isolation_forest"]
    
    predictions = {}
    for name in model_names:
        try:
            with open(f"models/{name}_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            if name == "isolation_forest":
                # Isolation Forest returns -1/1, convert to 0/1 anomaly scores
                preds = model.decision_function(X_test)
                # Normalize to [0,1] as anomaly score
                proba = 1 - (preds - preds.min()) / (preds.max() - preds.min())
                y_pred = (proba > 0.5).astype(int)
            else:
                try:
                    proba = model.predict_proba(X_test)[:, 1]
                except:
                    proba = model.predict(X_test)
                y_pred = (proba > 0.5).astype(int)
            
            predictions[name] = {
                "proba": proba,
                "pred": y_pred,
            }
            
            # Compute metrics
            metrics = compute_all_metrics(y_test, proba, y_pred)
            print(f"\n{name.upper()}:")
            for metric, value in metrics.items():
                paper_value = PAPER_METRICS.get(name, {}).get(metric, "N/A")
                print(f"  {metric}: {value:.4f} (paper: {paper_value})")
            
        except FileNotFoundError:
            print(f"Model {name} not found, skipping...")
            continue
    
    # Bootstrap CIs for LightGBM
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (B=1,000)")
    print("="*60)
    
    if "lightgbm" in predictions:
        lgb_proba = predictions["lightgbm"]["proba"]
        lgb_pred = predictions["lightgbm"]["pred"]
        
        bootstrap_results = bootstrap_metrics(y_test, lgb_proba, lgb_pred, 
                                             n_iterations=1000, seed=456)
        
        print("\nLightGBM with 95% CI:")
        for metric in ["roc_auc", "pr_auc", "f1", "accuracy", 
                       "precision", "recall", "mcc"]:
            mean = bootstrap_results[f"{metric}_mean"]
            std = bootstrap_results[f"{metric}_std"]
            ci_lower = bootstrap_results[f"{metric}_ci_lower"]
            ci_upper = bootstrap_results[f"{metric}_ci_upper"]
            paper_val = PAPER_METRICS["lightgbm"][metric]
            print(f"  {metric:12s}: {mean:.4f} ± {std:.4f} "
                  f"(95% CI: {ci_lower:.4f}–{ci_upper:.4f}) "
                  f"[paper: {paper_val}]")
    
    # Statistical significance tests
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    print(f"Bonferroni-corrected threshold: α_adj = 0.05/7 = {0.05/7:.4f}")
    
    if "lightgbm" in predictions:
        lgb_proba = predictions["lightgbm"]["proba"]
        
        for baseline in ["xgboost", "random_forest", "graphsage", 
                        "mlp", "gat", "logistic_regression", 
                        "isolation_forest"]:
            if baseline in predictions:
                result = paired_significance_test(
                    y_test, lgb_proba, predictions[baseline]["proba"], baseline
                )
                
                sig_mark = "***" if result["significant"] else ""
                print(f"\nLightGBM vs {baseline}:")
                print(f"  Δ AUC: {result['delta_auc']:+.4f}")
                print(f"  t-statistic: {result['t_statistic']:+.2f}")
                print(f"  p-value (two-sided): {result['p_value_two_sided']:.4f}")
                print(f"  Wilcoxon p: {result['wilcoxon_p']:.4f}")
                print(f"  Significant (α_adj={result['bonferroni_threshold']:.4f}): "
                      f"{result['significant']} {sig_mark}")


if __name__ == "__main__":
    main()
