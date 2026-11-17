# COMPREHENSIVE LITERATURE REVIEW: KEYSTROKE DYNAMICS FOR USER AUTHENTICATION

**Project:** Machine Learning-Based User Authentication Using Keystroke Dynamics  
**Course:** CSL236 – Introduction to AI & Machine Learning, B.Tech CSE, Sem 5, The NorthCap University  
**Dataset Reference:** Carnegie Mellon University (CMU) Keystroke Dynamics Benchmark (`DSL-StrongPasswordData.csv`)

---

## 1. Executive Summary of Research Context

Keystroke dynamics is a behavioral biometric that authenticates a user based on their unique typing rhythm: how long they press each key and how quickly they transition between consecutive keys. Unlike physical biometrics like fingerprints or face scans, keystroke dynamics works directly on standard computer keyboards with zero extra hardware.

To design our machine learning system properly, we conducted a systematic literature review of five foundational and recent peer-reviewed papers spanning 2009 to 2026. For each paper, we broke down the core problem, the methodology, the key experimental findings, and how it directly guided our preprocessing, model selection, threshold calibration, and production monitoring.

---

## 2. Detailed Systematic Review of Core References

### Paper 1: Killourhy and Maxion (2009)
* **Full Citation:** Killourhy, K. S., & Maxion, R. A. (2009). Comparing Anomaly-Detection Algorithms for Keystroke Dynamics. In *2009 IEEE/IFIP International Conference on Dependable Systems & Networks (DSN)* (pp. 125–134). IEEE.
* **DOI / Link:** [https://doi.org/10.1109/DSN.2009.5270346](https://doi.org/10.1109/DSN.2009.5270346)
* **Research Problem:** Lack of standardized benchmarks and evaluation methodologies across keystroke biometric literature. Prior studies utilized disparate passwords, unreleased datasets, varying feature definitions, and inconsistent train/test splits, preventing objective cross-algorithm comparison.
* **Methodology:** 
  - Created the seminal **CMU Keystroke Dynamics Benchmark Dataset** (`DSL-StrongPasswordData.csv`), collecting 20,400 samples from 51 subjects typing the complex password `.tie5Roanl` 400 times across 8 sessions spaced at least 24 hours apart.
  - Extracted 31 temporal features per sample: 11 Hold times ($H$), 10 Keydown-to-Keydown latencies ($DD$), and 10 Keyup-to-Keydown intervals ($UD$).
  - Evaluated 14 anomaly detection algorithms (including Manhattan distance, Euclidean distance, Mahalanobis distance, One-Class SVM, Nearest Neighbor, and Neural Networks) using Equal Error Rate (EER) and ROC analysis.
  - Standardized split: 200 training repetitions per subject vs 200 testing repetitions (with 50 other subjects acting as zero-effort impostors).
* **Key Findings:**
  - Simple distance-based anomaly detectors often outperformed complex machine learning algorithms. Specifically, Manhattan (filtered) distance achieved the best overall benchmark EER of **9.6%**, followed by Outlier Count (10.2%) and Mahalanobis distance (11.0%).
  - One-Class SVM achieved 10.2% EER, while multi-layer perceptron neural networks suffered from overfitting under limited enrollment samples.
  - High inter-session temporal variability (typing fatigue, emotional state, time of day) accounted for significant degradation between earlier and later sessions.
* **Relevance to Our Project:**
  - This paper created the exact dataset used in our project (`DSL-StrongPasswordData.csv`).
  - Directly justifies our 31-dimensional feature schema ($H$, $DD$, $UD$).
  - Explains why session-based splitting (Sessions 1–4 Train vs Sessions 5–8 Test) is mandatory to prevent optimistic data leakage.
  - Provides the canonical benchmark EER (~9.6% to 10.2%) against which our models are evaluated.
* **Identified Limitations:**
  - Focuses strictly on anomaly detection (one-class training) without evaluating supervised discriminative models (e.g., modern Random Forest or Gradient Boosting).
  - Assumes a static 10-character password, limiting findings to fixed-text authentication.

---

### Paper 2: Teh, Teoh, and Yue (2013)
* **Full Citation:** Teh, P. S., Teoh, A. B. J., & Yue, S. (2013). A Survey of Keystroke Dynamics Biometrics. *The Scientific World Journal*, 2013, Article ID 408280, 24 pages.
* **DOI / Link:** [https://doi.org/10.1155/2013/408280](https://doi.org/10.1155/2013/408280)
* **Research Problem:** Comprehensive taxonomy and comparative review of keystroke dynamics processing pipelines, encompassing data acquisition, preprocessing, feature extraction, classification paradigms, and spoofing vulnerabilities.
* **Methodology:** 
  - Surveyed over 100 keystroke dynamics publications across two decades.
  - Categorized systems along three axes: input modalities (fixed-text password vs free-text continuous typing), timing feature formulations (digraphs, trigraphs, flight time, dwell time), and decision engines (statistical distance metrics, supervised classifiers, unsupervised anomaly detectors).
  - Investigated biometric security vulnerabilities including synthetic keystroke injection, statistical profiling attacks, and template protection.
* **Key Findings:**
  - Fixed-text authentication consistently yields lower error rates (EER 2%–10%) than free-text verification (EER 10%–25%) due to rigid grammatical and motor-pattern constraints.
  - Negative Keyup-to-Keydown ($UD$) timing values occur systematically in fast typists due to key rollover (depressing the next key prior to releasing the previous one); algorithms that truncate or clip negative timings artificially distort the biometric manifold.
  - Classifier choice must account for severe class imbalance in single-user authentication ($1 \text{ genuine} : N \text{ impostors}$).
* **Relevance to Our Project:**
  - Provides mathematical and ergonomic justification for preserving negative $UD$ timing values in our data contract and preprocessing pipeline.
  - Validates our fixed-text problem formulation as an industry-standard secondary authentication layer.
  - Emphasizes the necessity of reporting FAR, FRR, and EER rather than naive accuracy.
* **Identified Limitations:**
  - Pre-dates modern deep representation learning (e.g., Siamese and Triplet networks) and automated MLOps continuous monitoring frameworks.

---

### Paper 3: Pisani and Lorena (2013)
* **Full Citation:** Pisani, P. H., & Lorena, A. C. (2013). A systematic review on keystroke dynamics. *Pattern Analysis and Applications*, 16(4), 473–495.
* **DOI / Link:** [https://doi.org/10.1007/s13173-013-0117-7](https://doi.org/10.1007/s13173-013-0117-7)
* **Research Problem:** Systematic literature review evaluating feature representation methods, dimensionality reduction, and evaluation protocols to address high dimensionality and high intraclass variability in keystroke biometrics.
* **Methodology:**
  - Systematic review protocol analyzing feature selection methods (Filter, Wrapper, Embedded), feature extraction methods (PCA, LDA), and classification algorithms across diverse public benchmark databases.
  - Rigorous examination of evaluation metrics: False Match Rate (FMR/FAR), False Non-Match Rate (FNMR/FRR), Receiver Operating Characteristic (ROC) curves, and Area Under the Curve (AUC).
* **Key Findings:**
  - High correlation exists among adjacent transition features (e.g., $DD$ and $UD$ pairs for the same character transition: $DD = H + UD$).
  - Principal Component Analysis (PCA) successfully reduces feature space multicollinearity while retaining 90%–95% of total variance, mitigating curse of dimensionality without sacrificing discriminative power.
  - Data leakage was identified in several reviewed studies where scalers or PCA projections were fitted over the entire dataset prior to cross-validation or train/test splitting, producing artificially optimistic EERs.
* **Relevance to Our Project:**
  - Directly informs our use of `StandardScaler` followed by `PCA(0.95)` as established in `reference/Project_Aiml.ipynb`.
  - Enforces our strict pipeline design: `fit()` only on training sessions `[1, 2, 3, 4]`, and `transform()` on testing sessions `[5, 6, 7, 8]`.
  - Clarifies the mathematical collinearity relationship: $DD.k1.k2 \approx H.k1 + UD.k1.k2$.
* **Identified Limitations:**
  - Did not investigate tree-based ensemble methods (e.g., Random Forest, XGBoost) on non-linear decision boundaries.

---

### Paper 4: Teotia, Sharma, and Elmiligi (2023)
* **Full Citation:** Teotia, V., Sharma, S., & Elmiligi, H. (2023). A comprehensive study of machine learning approaches for keystroke dynamics authentication. *International Journal of Biometrics*, 15(2), 148–172.
* **DOI / Link:** [https://doi.org/10.1504/IJBM.2023.133148](https://doi.org/10.1504/IJBM.2023.133148)
* **Research Problem:** Comparative benchmarking of modern machine learning algorithms (Logistic Regression, Support Vector Machines, Random Forest, k-Nearest Neighbors, and Multi-Layer Perceptrons) on fixed-text benchmark datasets with emphasis on class imbalance and decision threshold calibration.
* **Methodology:**
  - Conducted extensive empirical experiments on the CMU benchmark dataset (`DSL-StrongPasswordData.csv`).
  - Evaluated the impact of extreme class imbalance ($1 : 50$) on standard supervised classifiers.
  - Applied synthetic minority oversampling (SMOTE), class weighting, and decision threshold sweeps to calibrate classifiers for biometric verification.
* **Key Findings:**
  - Default supervised classifiers (e.g., unweighted Logistic Regression at default 0.5 threshold) suffer catastrophic False Rejection Rates ($FRR > 95\%$) despite achieving $>97\%$ accuracy, because the model collapses toward the majority impostor class.
  - Random Forest and Support Vector Machines with RBF kernel consistently achieve superior ROC-AUC ($>0.96$) compared to linear baselines.
  - Tuning the operating threshold to the Equal Error Rate (EER) or applying cost-sensitive loss functions (`class_weight='balanced'`) restores operational authentication balance, bringing both FAR and FRR below 5%–10%.
* **Relevance to Our Project:**
  - Directly explains and mathematically validates the exact baseline failure mode observed in our midterm notebook (`reference/Project_Aiml.ipynb`):
    $$\text{Accuracy} = 97.90\%, \quad \text{Recall} = 0.0100, \quad \text{FRR} = 99.00\%, \quad \text{FAR} = 0.16\%$$
  - Directly justifies comparing Logistic Regression against Random Forest and SVM in our MLflow experiments (`train.py`).
  - Demonstrates why threshold optimization and probability calibration are essential for production deployment in `app.py`.
* **Identified Limitations:**
  - Focused purely on offline model evaluation without addressing API serving, containerization, or continuous monitoring for production drift.

---

### Paper 5: Subash et al. (2026)
* **Full Citation:** Subash, A., Ghosh, R., Debbarma, K., & Bhowmik, M. K. (2026). Adaptability of current keystroke and mouse behavioral biometric systems: A survey. *Computers & Security*, 150, Article 104731.
* **DOI / Link:** [https://doi.org/10.1016/j.cose.2025.104731](https://doi.org/10.1016/j.cose.2025.104731)
* **Research Problem:** Investigating the adaptability, temporal stability, and production longevity of behavioral biometrics (keystroke and mouse dynamics) in live enterprise deployment environments.
* **Methodology:**
  - Systematic examination of longitudinal behavioral biometric studies.
  - Analyzed covariate factors inducing feature distribution drift: user fatigue, cognitive workload, hardware variations (mechanical vs membrane vs laptop keyboards), temporal aging of motor templates, and physical health changes.
  - Evaluated MLOps monitoring paradigms, drift detection algorithms (Population Stability Index, Kolmogorov-Smirnov test), and model retraining triggers.
* **Key Findings:**
  - Keystroke dynamics models experience concept drift and covariate shift over time; models evaluated across sessions spaced weeks apart exhibit an average EER increase of 4%–12% if templates are not adapted.
  - Inter-session variability is significantly higher than intra-session variability, proving that single-session evaluations vastly underestimate real-world error rates.
  - Robust production deployments require runtime input logging and automated drift detection comparing inference distributions against enrollment baselines to trigger alerts and retraining.
* **Relevance to Our Project:**
  - Provides theoretical and practical foundation for our **Bonus Feature B: Data Drift Monitoring** (`monitoring/drift_check.py`).
  - Justifies our prediction logging implementation (`monitoring/prediction_logs.jsonl`) to track production inference statistics.
  - Supports our 3-way session-based evaluation protocol (training on Sessions 1–3, validating on Session 4, and testing on unseen Sessions 5–8) to evaluate model resilience across time.
* **Identified Limitations:**
  - Highlights that privacy-preserving logging standards must avoid storing raw typed content or reconstructed passwords, focusing strictly on statistical timing distributions.

---

## 3. Synthesis: Mapping Literature Findings to Our End-Term Implementation

| Literature Finding | Academic Source | How Our Project Implements It |
|---|---|---|
| **Feature Representation ($H, DD, UD$)** | Killourhy & Maxion (2009) | 31 exact timing features extracted from `.tie5Roanl` password input. |
| **Negative $UD$ Preservation** | Teh, Teoh, & Yue (2013) | Documented in data contract; negative values preserved as physical key rollover. |
| **Data Leakage & Multicollinearity** | Pisani & Lorena (2013) | `StandardScaler` + `PCA(0.95)` fitted strictly on training data (Sessions 1–3 for validation tuning; Sessions 1–4 for final production serialization). |
| **The 99% FRR Baseline Trap & EER Tuning** | Teotia et al. (2023) | Documented in `train.py` & MLflow; baseline LR contrasted with balanced RF/SVM and EER threshold calibration. |
| **Temporal Shift & Drift Monitoring** | Subash et al. (2026) | Sessions 5–8 held out for testing; runtime logging & `monitoring/drift_check.py` implemented. |
