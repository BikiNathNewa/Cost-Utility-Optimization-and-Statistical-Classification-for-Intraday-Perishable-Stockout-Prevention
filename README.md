# MAS Project
# Cost-Utility Optimization and Statistical Classification for Intraday Perishable Stockout Prevention


[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Jupyter Notebooks](https://img.shields.io/badge/Notebooks-Pre--Rendered-orange.svg)]()
[![Status: Complete](https://img.shields.io/badge/Status-Fully%20Implemented-success.svg)]()

---

## 1. Primary Jupyter Notebook Deliverables

| Notebook File | Focus & Criteria | Key Outputs & Analyses |
| :--- | :--- | :--- |
| **[`stockout_classifier_pipeline.ipynb`](stockout_classifier_pipeline.ipynb)** | **Full Pipeline (Criteria 1 to 8)** | Complete end-to-end master notebook combining data extraction, EDA, distributional fitting, hypothesis testing, logistic regression, kNN, and cost-utility threshold optimization. |
| **[`01_eda_and_distributional_modeling.ipynb`](01_eda_and_distributional_modeling.ipynb)** | **Phase 1 & 2 (Criteria 1 to 4)** | Problem definition, top 5 cities filter, descriptive statistics, parallel box plots, diurnal rhythm, Gaussian MLE, K-S tests, Q-Q diagnostics, and 1,000 bootstrap resamples. |
| **[`02_hypothesis_testing_and_classification.ipynb`](02_hypothesis_testing_and_classification.ipynb)** | **Phase 3 & 4 (Criteria 5 to 8)** | Levene's homoscedasticity test, Welch's two-sample t-test, Mann-Whitney U rank-sum test, Cohen's d effect size, Logistic Regression, kNN, and cost-optimal decision threshold tuning. |
| **[`dashboard/index.html`](dashboard/index.html)** | **Interactive Dashboard Console** | Production-grade web console built strictly under `ui-ux-pro-max-skill` design tokens. Features live threshold simulator, cost curve analytics, ROC/PR curves, 5-fold CV matrix, and live SKU rebalance dispatch feed. |

---

## 2. Project Overview & Operational Decision Frame (Criterion 1)
* **Decision Maker:** Regional Inventory Dispatcher at Dingdong Fresh.
* **Decision Action:** Trigger an emergency intra-day inventory replenishment transfer from regional fulfillment centers to local frontline micro-fulfillment warehouses (FMWs) before 12:00 PM for high-risk SKUs.
* **Specific Research Question:** *"Does the 06:00 to 10:00 morning sales pace reliably predict whether an SKU will stock out before 18:00 across the top 5 metropolitan cities, allowing dispatchers to rebalance stock mid-day?"*
* **Operational Asymmetric Cost Matrix:**
  * Transfer Cost (False Positive transfer): $15.0\text{ RMB}$ (van transit and handling).
  * Stockout Cost (False Negative failure): $50.0\text{ RMB}$ (lost margin and customer churn).
  * Bayes Optimal Decision Threshold: $p^* = \frac{15}{15 + 50} \approx 0.23$.

---

## 3. Repository Structure
```
MMAS/
├── app.py                                         # Production Flask Backend & ML Inference Server
├── dashboard/                                     # Fully Functional Interactive Web Dashboard
│   ├── index.html                                 # Single-page operational console (Tailwind & Chart.js)
│   ├── data.js                                    # Bundled verified tables, curves, and 60 SKU alerts
│   └── README.md                                  # Dashboard architecture and launching instructions
├── 01_eda_and_distributional_modeling.ipynb       # Phase 1 & 2 
├── 02_hypothesis_testing_and_classification.ipynb # Phase 3 & 4
├── stockout_classifier_pipeline.ipynb             # Master Unified Notebook
├── Course Content/                                # Reference material and project rubrics
├── data/
│   ├── train.parquet                              # FreshRetailNet-50K raw train partition (4.5M rows)
│   └── eval.parquet                               # Evaluation partition (350K rows)
├── src/                                           # Supporting library modules
│   ├── config.py                                  # Seeds, parameters, city IDs, costs
│   ├── data_loader.py                             # Vectorized chunk extraction and filtering
│   ├── eda.py                                     # Descriptive statistics and plotting
│   ├── distributional_modeling.py                 # Gaussian MLE, KS tests, Q-Q plots
│   ├── estimation.py                              # Student's t bounds and bootstrap CIs
│   ├── hypothesis_testing.py                      # Welch's t, Mann-Whitney U, Cohen's d
│   ├── modeling.py                                # Baselines, Logit, kNN, Cost optimization
│   └── utils.py                                   # Styling and table formatting
├── results/
│   ├── figures/                                   # Exported 300 DPI figures
│   └── tables/                                    # Exported CSV and JSON summary tables
├── run_pipeline.py                                # Reproducible CLI script (~21 seconds)
├── DATA_CARD.md                                   # Structured Dataset Card
├── REPORT.md                                      # Full 3-4 page academic report
├── requirements.txt                               # Pinned dependency specifications
```

---

## 4. Key Methodological Decisions & Alternatives

| Method Used | Why Decided & How Chosen | Limitations of Method | Alternatives Considered & Why Rejected |
| :--- | :--- | :--- | :--- |
| **Top 5 Cities Filter** (`city_id in [0, 12, 16, 3, 6]`) | Isolates dense metropolitan markets where on-demand 30-minute delivery and intraday van rebalancing are active. Retains 3,643,920 records (80.98%). | Excludes rural or small suburban hubs with next-day logistics. | *Full dataset:* Rejected due to sparse order patterns distorting intraday pacing. *City 0 alone:* Rejected to ensure cross-city generalizability. |
| **Outlier Robustness (Winsorized & Trimmed Means, MAD)** | Evaluated whether max morning sales (17.3 units) distorted the +0.106 mean gap. 1% and 5% Winsorized differences (+0.103 and +0.099) and 5% trimmed difference (+0.088) confirm an authentic location shift. | Symmetrically clips legitimate high-volume customer orders. | *Raw mean alone:* Sensitive to extreme tail spikes. *Outlier removal:* Discarding points violates empirical data provenance. |
| **Distributional Modeling with Multiplicity Correction** | Fit Gaussian via MLE; tested via one-sample and two-sample K-S tests. Enforced Bonferroni correction ($m=3, \alpha_{\text{adj}} = 0.0167$) to strictly control Family-Wise Error Rate (all $p_{\text{adj}} < 10^{-300}$). | K-S test has near-infinite power in massive sample sizes ($N > 10^6$). | *Uncorrected K-S tests:* Rejected due to FWER inflation (14.3%). *Poisson:* Rejected due to severe overdispersion ($\text{Var} \gg \text{Mean}$). |
| **Estimation (Student's t & Bootstrap CIs)** | Student's t for parametric baseline; 1,000-resample empirical bootstrap for distribution-free CI on mean difference ($[0.1048, 0.1069]$). | Student's t assumes normality; bootstrap is computationally heavier. | *Normal z-interval:* Rejected because population variance is unknown. *Bayesian Credible Interval:* Rejected to preserve frequentist design. |
| **Hypothesis Testing (Welch's t & Mann-Whitney U)** | Welch's t-test handles unequal variances ($s_1^2 / s_0^2 = 1.802$); Mann-Whitney U test operates on ranks to protect against heavy right-tail outliers. | Welch's t relies on CLT; Mann-Whitney tests stochastic rank dominance rather than mean difference. | *Student's standard t-test:* Rejected due to homoscedasticity failure. *Permutation test:* Conceptually identical to bootstrap but slower on $N > 3.6\times 10^6$. |
| **5-Fold Cross-Validation & kNN Tuning** | Quantified performance uncertainty ($\text{Mean} \pm \text{SD}$) via 5-fold Stratified CV (AUC SD $\le 0.0054$). Evaluated $k \in \{5, 10, 15, 20\}$ to empirically justify $k=15$. | Adds training runtime overhead across 20 fit iterations. | *Single split alone:* Vulnerable to partition bias. *Arbitrary k choice:* Heuristic selection lacks empirical justification. |
| **Class Imbalance: Bayes Threshold Tuning vs Balanced Weights** | Post-hoc Bayes threshold tuning ($p^* = 0.24$) optimizes for the $3.33\times$ cost ratio while maintaining probability calibration (Brier loss 0.179 vs 0.201 for balanced LR). | Requires accurate domain estimates of unit error costs ($C_{\text{FP}}, C_{\text{FN}}$). | *class_weight='balanced':* Distorts posterior probabilities and over-penalizes false negatives compared to empirical business costs. |
| **Decision Optimization ($p^* = 0.24$)** | Minimizes expected business failure costs ($C_{\text{transfer}} = 15$ vs $C_{\text{stockout}} = 50$). 1,000 bootstrap test resamples confirm 26.38% cost reduction (95% CI: $[24.34\%, 28.34\%]$). | Assumes static depot replenishment capacity. | *Standard cutoff ($p = 0.50$):* Rejected because it produces poor recall (26.3%) by treating false alarms and missed stockouts as equally costly. |

---

