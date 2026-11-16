# Dataset Card: FreshRetailNet-50K (Top 5 Cities Intraday Stockout Subset)

## 1. Dataset Summary & Provenance
* **Dataset Name:** FreshRetailNet-50K
* **Origin / Publisher:** Dingdong Fresh (`Dingdong-Inc`) / Dingdong Research
* **Repository:** [Hugging Face (`Dingdong-Inc/FreshRetailNet-50K`)](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) and [GitHub (`Dingdong-Inc/frn-50k-baseline`)](https://github.com/Dingdong-Inc/frn-50k-baseline)
* **Dataset Domain:** On-demand fresh grocery retail (e-grocery) operating via frontline micro-fulfillment warehouses (FMWs) in major metropolitan cities.
* **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Permitted for academic, research, and commercial applications with proper attribution.
* **Citation:** Dingdong Fresh Research Team, *FreshRetailNet-50K: A Large-Scale Real-World Benchmark for Censored Demand and Stockout Modeling in E-Grocery*, 2024.

---

## 2. Research Objective & Contextual Framing
In rapid on-demand fresh retail, perishables (vegetables, fruits, meats, dairy) are stocked in local urban micro-warehouses to fulfill 30-minute delivery promises. Stockouts during peak afternoon and dinner hours (10:00-18:00) cause immediate lost sales, stockout penalty costs, and customer churn. 

This dataset subset is configured to evaluate whether early morning sales pace ($06:00\text{--}10:00$) reliably predicts mid-day stockout occurrence ($10:00\text{--}18:00$), enabling regional inventory dispatchers to trigger emergency intra-day replenishments before 12:00 PM.

---

## 3. Dataset Structure & Filtering Steps

### Raw Dataset Scale
* **Raw Records in Training Partition:** 4,500,000 store-product daily time series records.
* **Temporal Coverage:** 90 calendar days from March 28, 2024 to June 25, 2024.
* **Evaluation Partition:** 350,000 records from June 26, 2024 to July 02, 2024.

### Pre-Processing & Filtering Decisions (Criterion 2 & 7)
1. **Geographic Filtering:** Filtered raw records to the top 5 cities by transaction and order volume: `city_id in [0, 12, 16, 3, 6]`.
   * **Retained Sample Size:** 3,643,920 records (80.98% of the total dataset).
   * **Active Stores:** 630 frontline micro-warehouses.
   * **Monitored SKUs:** 686 unique fresh food product items.
2. **Missingness & Outlier Auditing:**
   * An audit across all 19 columns identified **0 null or missing values** across all 4,500,000 raw rows.
   * No records were artificially discarded to force normality; the extreme right tail represents legitimate bulk purchasing or restaurant/office demand spikes.
3. **Engineered Features & Operational Definitions:**
   * **Predictor Variable ($X_1$ - Morning Sales Pace):**
     $$\text{Sales}_{06-10} = \sum_{h=6}^{9} \text{hours\_sale}[h]$$
     Sum of customer purchase units during the morning operating window (06:00 to 10:00 AM).
   * **Predictor Variable ($X_2$ - Morning Stockout Status):**
     $$\text{StockSum}_{06-10} = \sum_{h=6}^{9} \text{hours\_stock\_status}[h]$$
     Count of hours during the morning window where the SKU was already depleted.
   * **Target Outcome Variable ($Y_i$):**
     $$Y_i = \mathbb{I}\left(\sum_{h=10}^{17} \text{hours\_stock\_status}[h] > 0\right)$$
     Binary indicator: $Y_i = 1$ if the SKU experiences at least one hour of stockout between 10:00 and 18:00; $Y_i = 0$ if in-stock continuously.
   * **Contextual Exogenous Covariates:**
     * `discount`: Promotional discount ratio applied to the SKU ($0.0$ to $1.0$).
     * `precpt`: Total precipitation in millimeters (influences rainy-day order surges).
     * `avg_temperature`: Mean local ambient temperature in Celsius (influences perishable spoil and cold beverage/produce demand).
     * `holiday_flag`: Binary indicator ($1$ = statutory public holiday or weekend festival, $0$ = regular day).

---

## 4. Key Summary Statistics
| Metric | Non-Stockout ($Y=0$) | Stockout ($Y=1$) | Combined Population |
| :--- | :--- | :--- | :--- |
| **Record Count ($N$)** | 2,465,214 (67.65%) | 1,178,706 (32.35%) | 3,643,920 (100.0%) |
| **Morning Sales Mean ($\mu$)** | 0.2540 units | 0.3599 units | 0.2883 units |
| **Morning Sales Std ($\sigma$)** | 0.3932 units | 0.5278 units | 0.4440 units |
| **Morning Sales Median** | 0.2000 units | 0.2000 units | 0.2000 units |
| **Morning Sales IQR** | 0.3000 units | 0.4000 units | 0.4000 units |
| **Zero-Sales Incidence** | 26.25% | 23.05% | 25.21% |
| **Skewness** | +6.568 | +5.835 | +6.355 |
| **Kurtosis** | 90.32 | 69.71 | 83.96 |

---

## 5. Data Privacy, Anonymization & Ethical Considerations (Criterion 7)
* **Personally Identifiable Information (PII):** None. The dataset contains zero customer identifiers, consumer payment records, delivery driver names, or vehicle telematics.
* **Hashed Identifiers:** `store_id`, `product_id`, `management_group_id`, and `city_id` are anonymized synthetic integer hashes preventing the re-identification of proprietary store addresses or secret pricing schedules.
* **Analytical Transparency:** All records from the top 5 cities are preserved without selective exclusion. Outliers are retained in all hypothesis tests and modeling routines to avoid false-positive inferential bias.

---

## 6. Known Limitations & Caveats
1. **Demand Censoring:** Observed sales drop to 0 whenever an SKU is out of stock. Morning sales pace $\text{Sales}_{06-10}$ for SKUs that stocked out before 10:00 AM represents *censored sales*, not uncensored latent customer demand. Including $\text{StockSum}_{06-10}$ explicitly models this censoring condition.
2. **Geographic Concentration:** Data reflects high-density top-tier Chinese metropolitan hubs with established 30-minute rapid delivery networks. Findings may not generalize to suburban or rural e-commerce models with next-day distribution.

