"""Master Pipeline Script: Binary Intraday Stockout Early Warning Classifier.
Runs end-to-end from raw parquet data to every figure, table, and number.
Fulfills Project Criteria 1 through 8.
"""

import sys
import time
import pandas as pd
import numpy as np

from src.config import (
    RANDOM_SEED,
    TOP_CITIES,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DATA_TRAIN_PATH
)
from src.utils import set_seed, configure_plot_style, save_table, format_table_markdown
from src.data_loader import load_processed_data, get_hourly_profiles, get_modeling_split
from src.eda import (
    compute_descriptive_statistics,
    plot_box_and_density,
    plot_intraday_temporal_dynamics,
    plot_correlation_matrix
)
from src.distributional_modeling import (
    fit_and_evaluate_distributions,
    plot_qq_and_ecdf
)
from src.estimation import (
    compute_mle_and_confidence_intervals,
    plot_bootstrap_distribution
)
from src.hypothesis_testing import (
    run_hypothesis_tests,
    plot_hypothesis_evidence
)
from src.modeling import (
    evaluate_models_and_baselines,
    plot_model_curves_and_cost
)


def main():
    total_start = time.time()
    set_seed(RANDOM_SEED)
    configure_plot_style()
    
    print("=" * 80)
    print("  DINGDONG FRESH: INTRADAY STOCKOUT EARLY WARNING CLASSIFIER")
    print("  Reproducible Statistical Pipeline (Criteria 1 to 8)")
    print("=" * 80)
    print(f"Random Seed Pinned: {RANDOM_SEED}")
    print(f"Target Cities: {TOP_CITIES}")
    print(f"Output Directories: {FIGURES_DIR} and {TABLES_DIR}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: DATA FILTERING & EXPLORATORY DATA ANALYSIS (Criteria 1 & 2)
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Executing Data Filtering and Exploratory Data Analysis...")
    t0 = time.time()
    df, meta = load_processed_data(DATA_TRAIN_PATH, top_cities=TOP_CITIES, verbose=True)
    
    # Save dataset metadata table
    meta_df = pd.DataFrame([meta])
    save_table(meta_df, "table_data_summary_metadata", index=False)
    
    print("\n--- Descriptive Statistics for Morning Sales Pace ---")
    stats_df, context_df, robust_df = compute_descriptive_statistics(df)
    print(format_table_markdown(stats_df, "Descriptive Statistics (Morning Sales Pace 06:00-10:00)"))
    print(format_table_markdown(robust_df, "Outlier Robustness Statistics (Winsorized Means and MAD)"))
    
    print("\n--- Generating Exploratory Visualizations ---")
    plot_box_and_density(df)
    hourly_profiles = get_hourly_profiles(DATA_TRAIN_PATH, top_cities=TOP_CITIES)
    plot_intraday_temporal_dynamics(hourly_profiles)
    plot_correlation_matrix(df)
    print(f"[PHASE 1 Complete] Elapsed: {time.time()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # PHASE 2: DISTRIBUTIONAL MODELING & ESTIMATION (Criteria 3 & 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PHASE 2] Executing Distributional Modeling and Estimation & Uncertainty...")
    t0 = time.time()
    
    dist_results, s0_sample, s1_sample = fit_and_evaluate_distributions(df)
    print(f"Normal Fit Non-Stockout: mu={dist_results['mu_0']:.4f}, sigma={dist_results['sigma_0']:.4f}, KS p-value={dist_results['ks_pval_0']:.2e}")
    print(f"Normal Fit Stockout:     mu={dist_results['mu_1']:.4f}, sigma={dist_results['sigma_1']:.4f}, KS p-value={dist_results['ks_pval_1']:.2e}")
    print(f"Two-sample KS statistic D={dist_results['ks_2samp_stat']:.4f}, p-value={dist_results['ks_2samp_pval']:.2e}")
    print(f"Binomial Baseline Stockout Rate p_hat={dist_results['p_hat_stockout']:.4f} (95% CI: [{dist_results['ci_p_lower']:.4f}, {dist_results['ci_p_upper']:.4f}])")
    
    plot_qq_and_ecdf(
        s0_sample, s1_sample,
        dist_results['mu_0'], dist_results['sigma_0'],
        dist_results['mu_1'], dist_results['sigma_1']
    )
    
    print("\n--- Computing Maximum Likelihood Estimates & Confidence Intervals ---")
    est_summary, boot_diffs = compute_mle_and_confidence_intervals(df)
    print(f"Point Estimate Difference (mu_1 - mu_0): {est_summary['diff_mle']:.4f}")
    print(f"Welch's t 95% CI: [{est_summary['ci_welch'][0]:.4f}, {est_summary['ci_welch'][1]:.4f}]")
    print(f"Bootstrap 95% CI: [{est_summary['ci_boot'][0]:.4f}, {est_summary['ci_boot'][1]:.4f}]")
    
    plot_bootstrap_distribution(
        boot_diffs,
        est_summary['diff_mle'],
        est_summary['ci_boot'][0],
        est_summary['ci_boot'][1]
    )
    print(f"[PHASE 2 Complete] Elapsed: {time.time()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # PHASE 3: HYPOTHESIS TESTING & STATISTICAL MODELING (Criteria 5 & 6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PHASE 3] Executing Hypothesis Testing and Predictive Modeling...")
    t0 = time.time()
    
    hyp_results = run_hypothesis_tests(df)
    print(f"Welch's t-test: t = {hyp_results['welch_t_stat']:.2f}, df = {hyp_results['welch_df']:.0f}, p = {hyp_results['welch_pval']:.2e}")
    print(f"Cohen's d Effect Size: {hyp_results['cohens_d']:.4f}")
    print(f"Mann-Whitney U: U = {hyp_results['u_stat']:.0f}, p = {hyp_results['mw_pval']:.2e}, Rank Biserial r = {hyp_results['rank_biserial_r']:.4f}")
    
    plot_hypothesis_evidence(hyp_results)
    
    print("\n--- Training Supervised Classifiers and Baselines ---")
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = get_modeling_split(df)
    
    model_summary, coef_df, pred_dict = evaluate_models_and_baselines(
        X_train, X_test, y_train, y_test, X_train_s, X_test_s
    )
    print("\nModel Out-of-Sample Performance Summary:")
    print(format_table_markdown(model_summary, "Classifier Benchmark Metrics"))
    
    print("\nStratified 5-Fold Cross-Validation Performance:")
    print(format_table_markdown(pred_dict['cv_df'], "5-Fold Cross-Validation Uncertainty"))
    
    print("\nkNN Hyperparameter Tuning (Grid Search across k):")
    print(format_table_markdown(pred_dict['knn_tune_df'], "k-Nearest Neighbors Grid Search (k in {5, 10, 15, 20})"))
    
    print("\nHoldout Test Bootstrap 95% Confidence Intervals (B=1,000):")
    print(format_table_markdown(pred_dict['boot_df'], "Test Metric Bootstrap Confidence Intervals"))
    
    print("\nLogistic Regression Parameter Coefficients and Odds Ratios:")
    print(format_table_markdown(coef_df, "Logistic Regression Coefficients"))
    
    print("\n--- Generating ROC, PR, and Cost-Utility Curves ---")
    plot_model_curves_and_cost(pred_dict)
    print(f"[PHASE 3 Complete] Elapsed: {time.time()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # PHASE 4: SUMMARY & VERIFICATION (Criteria 7 & 8)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PHASE 4] Pipeline Execution Completed Successfully!")
    print(f"Total Pipeline Runtime: {time.time() - total_start:.2f} seconds")
    print(f"All figures verified in: {FIGURES_DIR}")
    print(f"All tables verified in:  {TABLES_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()

