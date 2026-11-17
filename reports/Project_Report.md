# Machine Learning-Based User Authentication Using Keystroke Dynamics

**Academic Project Report**  
**Course Code:** CSL236 — Introduction to Artificial Intelligence & Machine Learning  
**Degree Programme:** Bachelor of Technology in Computer Science & Engineering (Semester 5)  
**Department:** Department of Computer Science & Engineering, The NorthCap University  
**Academic Term:** End-Term Evaluation 2026  
**Student Roll No:** 23CSU001  
**Project Architecture Reference:** Brainmentors `mlops-starter` Production ML Blueprint  

---

## Abstract

Static passwords remain the primary authentication gate for most digital systems, but they are vulnerable to shoulder surfing, credential stuffing, and theft. Keystroke dynamics offers a non-intrusive behavioral biometric layer by analyzing *how* a user types rather than just *what* they type. 

In this project, we built an end-to-end machine learning system for user authentication using the Carnegie Mellon University (CMU) Keystroke Dynamics Benchmark dataset (`DSL-StrongPasswordData.csv`). The dataset contains 20,400 typing samples of a 10-character password (`.tie5Roanl`) collected from 51 subjects across 8 separate sessions. We enrolled subject `s002` as the genuine user (Class 1) and evaluated against the remaining 50 subjects as zero-effort impostors (Class 0). 

We implemented a leak-free 3-way temporal evaluation protocol (training on Sessions 1–3, tuning decision thresholds on Session 4 validation data, and testing on unseen Sessions 5–8). We analyzed the failure mode of the unweighted midterm baseline—which scored 97.90% accuracy but suffered a 99.50% False Rejection Rate due to severe 1:50 class imbalance—and resolved it using cost-sensitive class weighting and Equal Error Rate (EER) threshold calibration. 

Four model families were tracked in MLflow SQLite: Baseline Logistic Regression, Balanced Logistic Regression, Random Forest, and Support Vector Machines (SVM) with an RBF kernel. The final pipeline was packaged into `model/model.joblib`, deployed as a FastAPI REST service (`app.py`) with strict Pydantic range checks across all 31 timing features, validated using 6 automated Pytest tests, containerized via Docker (`python:3.11-slim`), and monitored for distribution shifts using Kolmogorov-Smirnov statistical tests.

---

## 1. Introduction

Security systems typically rely on three factors: what you know (passwords or PINs), what you have (smartcards or phone authenticators), and what you are (fingerprints, face recognition, or iris scans). While physical biometrics are accurate, they require specialized sensors, cause user friction, and cannot be revoked if a template is ever breached.

Keystroke dynamics belongs to the behavioral biometric category—what you *do*. It measures the subconscious rhythm of a person's typing: how long they hold each key (dwell or hold time) and how quickly they transition between keys (flight time). Because keystroke dynamics uses existing computer keyboards with zero additional hardware, it provides a practical, low-cost secondary authentication layer for websites, administrative portals, and developer environments.

---

## 2. Problem Statement

Given the timing measurements of a 10-character password entry (`.tie5Roanl`), our system must decide whether the attempt was made by the enrolled account owner (`s002`, Class 1) or by an impostor (`s003` through `s057`, Class 0).

In developing this system, we encountered three primary challenges:
1. **Severe Class Imbalance (1:50 Ratio):** In a single-user enrollment setup across 51 subjects, genuine attempts account for only ~1.96% of the data, while impostors account for 98.04%. Standard accuracy is misleading here because a naive model predicting "impostor" on every sample achieves 98.04% accuracy while rejecting every legitimate user attempt.
2. **Key Rollover and Negative $UD$ Values:** Fast typists often press the next key before releasing the previous one ($t_{\text{press}}(k+1) < t_{\text{release}}(k)$), resulting in negative Keyup-to-Keydown ($UD$) timing values. Naive data cleaning pipelines often mistake these for sensor errors and zero them out, destroying valuable biometric individuality.
3. **Temporal Motor Drift Across Sessions:** A person's typing rhythm naturally evolves over time due to practice, fatigue, and habituation. Randomly shuffling rows across sessions causes data leakage by allowing the model to memorize temporary quirks from a specific sitting.

---

## 3. Project Objectives

1. **Grounded Literature Review:** Study foundational and recent research papers on keystroke dynamics to justify our preprocessing, feature choices, and evaluation metrics.
2. **Exploratory Data Analysis (EDA):** Perform an in-depth audit of feature distributions, collinearity, key rollover physics, and ANOVA between-subject variance across all 31 timing features.
3. **Leakage-Free Preprocessing Pipeline:** Build a Scikit-learn pipeline using `StandardScaler` and `PCA` fitted strictly on training sessions.
4. **Reproduce and Resolve the Baseline Failure Mode:** Replicate the midterm uncalibrated baseline (97.90% accuracy, 99.50% FRR) and fix it through balanced loss weighting and Equal Error Rate (EER) threshold tuning.
5. **Experiment Tracking with MLflow:** Track multiple model architectures using MLflow with a persistent SQLite database (`sqlite:///mlflow.db`).
6. **Production Serving and Testing:** Package the best pipeline into a standalone file (`model/model.joblib`), serve it through FastAPI with strict Pydantic range validation, verify system behavior with automated Pytest tests, containerize with Docker, and build an automated drift monitor.

---

## 4. Literature Review

Our system design is guided by findings from five peer-reviewed papers:

1. **Killourhy and Maxion (2009):** Introduced the CMU Keystroke Dynamics Benchmark dataset (`DSL-StrongPasswordData.csv`). They evaluated 14 anomaly detectors across 51 subjects, proved that inter-session variance is significantly higher than intra-session variance, and established the standard benchmark Equal Error Rate range (~9.6%–10.2%).
2. **Teh, Teoh, and Yue (2013):** Surveyed keystroke dynamics biometrics, finding that fixed-text authentication provides much lower error rates than free-text typing. Crucially, they noted that negative Keyup-to-Keydown ($UD$) values are normal occurrences of key rollover in skilled typists, and warned against clipping them to zero.
3. **Pisani and Lorena (2013):** Evaluated feature representation and dimensionality reduction. They demonstrated that adjacent key timings exhibit high linear correlation ($DD \approx H + UD$) and showed that Principal Component Analysis (PCA) retaining 90%–95% variance eliminates multicollinearity without hurting biometric classification.
4. **Teotia, Sharma, and Elmiligi (2023):** Benchmarked supervised ML models under extreme 1:50 class imbalance. They proved that standard unweighted classifiers collapse toward the majority class and verified that tuning decision thresholds at the Equal Error Rate (EER) is necessary for practical deployment.
5. **Subash et al. (2026):** Investigated long-term behavioral biometric adaptation, demonstrating that typing patterns experience covariate drift over time and emphasizing the need for production inference logging and statistical drift checks (such as the two-sample Kolmogorov-Smirnov test).

---

## 5. Dataset Description & Keystroke Timing Physics

The CMU Keystroke Dynamics Benchmark consists of 20,400 rows and 34 columns. Each row records one typing repetition of the 10-character password `.tie5Roanl`. The columns comprise 3 metadata fields (`subject`, `sessionIndex`, `rep`) and 31 continuous timing features measured in seconds.

### The 31 Timing Features
The features represent three physical timing measurements:
* **Hold Times ($H$, 11 features):** How long each key is held down:
  $$H(k) = t_{\text{release}}(k) - t_{\text{press}}(k)$$
* **Keydown-to-Keydown ($DD$, 10 features):** The elapsed time between pressing key $k$ and pressing key $k+1$:
  $$DD(k_1, k_2) = t_{\text{press}}(k_2) - t_{\text{press}}(k_1)$$
* **Keyup-to-Keydown ($UD$, 10 features):** The interval between releasing key $k$ and pressing key $k+1$:
  $$UD(k_1, k_2) = t_{\text{press}}(k_2) - t_{\text{release}}(k_1)$$

### The Physics of Negative $UD$ Values (Key Rollover)
In rapid typing, a typist often depresses the next key before releasing the current key ($t_{\text{press}}(k_2) < t_{\text{release}}(k_1)$). In our dataset, **22,118 transitions (10.84% of all $UD$ entries) are negative**. 

For example, the transition from `i` to `e` (`UD.i.e`) has 4,811 negative values, reaching as low as $-0.2358$ seconds. Rather than being data corruption, these negative numbers reflect the fluid overlap of skilled typists. Our data validation schema deliberately accepts negative $UD$ values down to $-1.0$s.

---

## 6. Preprocessing & Feature Engineering

### A. Temporal Session-Based Splitting (Preventing Data Leakage)
To accurately mirror a real-world system where an employee enrolls on day one and logs in over subsequent weeks, we split data strictly across sessions:
* **Training Set:** Sessions 1, 2, and 3 ($N = 7,650$; 150 Genuine, 7,500 Impostor).
* **Validation Set:** Session 4 ($N = 2,550$; 50 Genuine, 2,500 Impostor) — used exclusively for threshold tuning and hyperparameter checks.
* **Production Serialization (Option B: Strictly Paired Pipeline):** The final serving model serialized into `model/model.joblib` retains the exact pipeline fitted on Sessions 1–3 paired directly with the validation threshold $\tau = 0.1094$ calibrated on Session 4. We deliberately avoid refitting on Sessions 1–4 to maintain a 100% mathematically paired calibration procedure, accepting the trade-off of using 150 genuine training repetitions instead of 200.
* **Unseen Holdout Test Set:** Sessions 5, 6, 7, and 8 ($N = 10,200$; 200 Genuine, 10,000 Impostor) — completely untouched until final evaluation.

A random row split (`train_test_split(shuffle=True)`) would leak repetitions from the same 15-minute sitting into both training and testing, producing unrealistically optimistic results.

### B. Scaling and Dimensionality Reduction
1. **StandardScaler:** Fitted strictly on the training set and applied downstream to validation and test sets.
2. **PCA (0.95 Variance):** Reduces the 31 timing features to **17 orthogonal principal components**. This eliminates the mathematical multicollinearity between digraph pairs ($DD \approx H + UD$) while preserving $95.2\%$ of the total timing variance.

---

## 7. Model Training, Baseline Failure Analysis, and MLflow Tracking

### Biometric Evaluation Metrics
Standard classification accuracy is unhelpful under 1:50 class imbalance. Instead, we evaluate:
* **False Acceptance Rate (FAR):** The percentage of impostor attempts falsely accepted:
  $$\text{FAR} = \frac{FP}{TN + FP}$$
* **False Rejection Rate (FRR):** The percentage of genuine attempts falsely rejected:
  $$\text{FRR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$$
* **Equal Error Rate (EER):** The operating point on the ROC curve where $\text{FAR} = \text{FRR}$. A lower EER indicates a better biometric classifier.

### Experimental Results in MLflow
All experiments were tracked in a local SQLite database (`sqlite:///mlflow.db`) under the experiment `Keystroke-Authentication`. Models were trained on Sessions 1–3, calibrated on Session 4 validation data, and tested on unseen Sessions 5–8 ($N = 10,200$):

| Run Name | Algorithm | Operating Threshold ($\tau$) | Accuracy | Precision | Recall | FAR | FRR | F1-Score | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|---|---|---|
| **01_Baseline_Logistic_Regression** | Logistic Regression (Unweighted) | 0.5000 | 97.90% | 0.0625 | 0.50% | **0.15%** | 99.50% | 0.0093 | 0.9017 | 17.00% |
| **02_Balanced_Logistic_Regression_EER** | Balanced Logistic Regression | 0.4445 | 89.79% | 0.1315 | **75.00%** | 9.91% | **25.00%** | 0.2237 | **0.9161** | **14.44%** |
| **03_Random_Forest_Tuned** *(Selected)* | Random Forest (100 trees, depth 6) | 0.1094 | 87.12% | 0.0702 | 45.50% | 12.05% | 54.50% | 0.1217 | 0.8582 | 20.03% |
| **04_SVM_RBF_Kernel** | Support Vector Machine (RBF Kernel) | 0.0071 | **94.57%** | **0.2154** | 67.00% | 4.88% | 33.00% | **0.3260** | 0.8223 | 22.91% |

### Discussion of Model Performance
1. **The Midterm Failure Mode Explained:** Run 1 replicates the midterm baseline. Because genuine attempts make up only 1.96% of the test data, the unweighted model learned to predict "impostor" almost every time. It achieved 97.90% accuracy, but rejected 199 out of 200 legitimate attempts ($FN=199$, $\text{FRR}=99.50\%$).
2. **Balanced Logistic Regression Restores Balance:** In Run 2, adding `class_weight='balanced'` and calibrating the decision threshold to the EER operating point on Session 4 validation data ($\tau = 0.4445$) restored genuine recall to **75.00%** with a **14.44% EER** and **0.9161 ROC-AUC** (the highest AUC in the study).

### Why Random Forest Was Selected for Production

We evaluated four experimental configurations across **three distinct algorithm families**:
1. **Linear Models (Logistic Regression):** Evaluated under both unweighted baseline and cost-sensitive balanced configurations.
2. **Decision Tree Ensembles (Random Forest):** 100 estimators, max depth 6, balanced class weighting.
3. **Kernel Methods (Support Vector Machines):** Radial Basis Function (RBF) kernel with balanced class weighting.

| Model Configuration | Algorithm Family | EER | ROC-AUC | FAR | FRR |
|---|---|---:|---:|---:|---:|
| Logistic Regression (Baseline) | Linear Model | 17.00% | 0.9017 | **0.15%** | 99.50% |
| Balanced Logistic Regression | Linear Model | **14.44%** | **0.9161** | 9.91% | **25.00%** |
| Random Forest *(Serving)* | Tree Ensemble | 20.03% | 0.8582 | 12.05% | 54.50% |
| Support Vector Machine (RBF) | Kernel Method | 22.91% | 0.8223 | 4.88% | 33.00% |

#### Trade-Off Analysis & Production Justification
* **Empirical Biometric Performance:** Objectively, Random Forest was **not** the best-performing model across raw biometric metrics. Balanced Logistic Regression achieved superior discrimination with the lowest EER (14.44%) and highest ROC-AUC (0.9161), while SVM achieved the lowest False Acceptance Rate (4.88%). Random Forest exhibited higher error rates (EER 20.03%, FRR 54.50%).
* **Explainability & Architecture Alignment:** Random Forest was retained as the production serving model because its decision tree structure natively provides Mean Decrease in Impurity (MDI) feature importances (`classifier.feature_importances_`). In our production service, these importances directly power the real-time diagnostic engine, explaining to the user exactly which keystrokes (e.g., `H.Shift.r` or `UD.i.e`) caused an authentication failure. Linear and kernel models require secondary post-hoc explainers (such as SHAP or permutation importance) that introduce significant latency overhead to the request cycle.
* **Holdout Integrity Disclosure:** Sessions 5–8 were strictly reserved for evaluation and were never used for threshold calibration or hyperparameter optimization. During project review, holdout results were inspected to verify algorithm behavior. We deliberately retained Random Forest to preserve the application's architectural explainability design rather than altering model selection post-hoc based on holdout metrics.

### Production Model Calibration Rationale (Option B)
For final serialization into `model/model.joblib`, we retain the exact Random Forest pipeline fitted on Sessions 1–3 alongside the threshold calibrated on Session 4 ($\tau = 0.1094$). We deliberately avoided refitting the model on Sessions 1–4 prior to deployment: incorporating Session 4 into training alters the tree splits and leaf probability distribution, invalidating the empirical assumption that $\tau = 0.1094$ remains the true EER threshold. Although this approach uses 150 genuine training repetitions rather than 200, it guarantees that the deployed model and its threshold are directly paired from an uncompromised validation procedure without extrapolation.

---

## 8. System Architecture & FastAPI Service

Our implementation follows the Brainmentors `mlops-starter` production architecture:

```
[Browser UI (static/index.html)] 
              │ (HTTP JSON POST)
              ▼
[FastAPI REST Service (app.py)]
    ├── Pydantic Input Validation (31 Range Bounds) ➔ HTTP 422 on Invalid Input
    ├── Standalone Pipeline (KeystrokeAuthPipeline in model/model.joblib)
    ├── Diagnostic Rule Engine (Hold Time & Flight Time Insights)
    ├── Asynchronous Logger ➔ monitoring/prediction_logs.jsonl
    └── Automated Drift Check ➔ monitoring/drift_check.py (Two-Sample KS Test)
```

### Endpoints
* `GET /health`: Returns service health, model type, enrolled user (`s002`), and active threshold.
* `POST /predict`: Ingests 31 timing features; returns boolean decision (`authenticated`), match score, threshold, and 3 diagnostic tips.
* `GET /`: Serves the interactive web interface.
* `GET /api/sample/genuine` & `GET /api/sample/impostor`: Provide authentic test samples for live demonstrations.

---

## 9. Automated Testing & Verification

We authored an automated test suite in `tests/test_app.py` using Pytest:
* `test_health_endpoint`: Confirms the health check returns HTTP 200 with model information.
* `test_ui_index_page`: Confirms the root route serves the HTML frontend.
* `test_domain_logic_genuine_vs_impostor`: Verifies the domain rule that an authentic `s002` sample scores higher than an impostor `s003` sample, successfully authenticating the genuine user while rejecting the impostor.
* `test_invalid_input_missing_field_returns_422`: Confirms that missing timing fields trigger HTTP 422 Unprocessable Entity.
* `test_boundary_validation_out_of_range_returns_422`: Confirms that out-of-range values (e.g., $H > 3.0$s) trigger HTTP 422.
* `test_negative_ud_rollover_accepted`: Confirms that authentic negative rollover timings are accepted (HTTP 200).

**Result:** All 6 tests pass cleanly (`6 passed in 8.52s`).

---

## 10. Containerization & CI/CD

### Lean Docker Container
Our `Dockerfile` uses `python:3.11-slim` and installs only the serving dependencies listed in `requirements-serve.txt` (omitting heavy training libraries like MLflow, Matplotlib, and Pytest). The container exposes port 8000, runs as a lightweight Uvicorn service, and includes a built-in health check.

### GitHub Actions CI
The CI workflow (`.github/workflows/ci.yml`) runs on every push to `main`:
1. Sets up Python 3.11 with pip caching.
2. Installs requirements.
3. Validates that modules import cleanly.
4. Executes a baseline training run to confirm reproducibility.
5. Runs the full Pytest suite.
6. Runs the data drift check.

---

## 11. Data Drift Monitoring & Privacy Considerations

### Statistical Drift Detection (`monitoring/drift_check.py`)
To monitor production behavior over time, inference requests are logged to `monitoring/prediction_logs.jsonl`. The drift script runs two-sample Kolmogorov-Smirnov (KS) tests comparing recent inference batches against the baseline enrollment distributions (Sessions 1–4). If more than 5 features show statistically significant drift ($p < 0.05$ and mean shift $> 15\%$), a drift alert is recorded in `monitoring/drift_report.json`.

### Privacy by Design
In accordance with ethical biometric guidelines, our system logs **only floating-point timing values and match scores**. Passwords, usernames, IP addresses, and personal identification information are never stored, logged, or exposed.

---

## 12. Limitations & Future Scope

### Limitations
1. **Keyboard Hardware Sensitivity:** Key travel distances and actuation forces differ between mechanical keyboards and laptop chiclet keyboards, affecting timing latencies.
2. **Fixed-Text Scope:** The system currently authenticates against a single fixed password (`.tie5Roanl`).

### Future Scope
1. **Continuous Free-Text Authentication:** Extending the model to continuously monitor typing during regular computer use (e.g., typing documents or emails) using rolling n-gram latencies.
2. **Siamese Neural Networks for Few-Shot Enrollment:** Using deep metric learning (as explored in our secondary research notebook, `reference/midterm_notebook.ipynb`) to enroll new users in real time without retraining the classifier.

---

## 13. Academic Integrity & AI Tools Disclosure

In compliance with NorthCap University academic standards, I declare that generative AI tools (such as Claude / DeepMind Antigravity) were used as an assistive tool for code scaffolding, documentation structure, and debugging. All data analysis, physical interpretations, experimental design, and conclusions were independently conducted, verified, and understood by the author.

---

## 14. References

1. Killourhy, K. S., & Maxion, R. A. (2009). Comparing Anomaly-Detection Algorithms for Keystroke Dynamics. In *2009 IEEE/IFIP International Conference on Dependable Systems & Networks (DSN)* (pp. 125–134). IEEE. [DOI: 10.1109/DSN.2009.5270346](https://doi.org/10.1109/DSN.2009.5270346).
2. Teh, P. S., Teoh, A. B. J., & Yue, S. (2013). A Survey of Keystroke Dynamics Biometrics. *The Scientific World Journal*, 2013, Article ID 408280. [DOI: 10.1155/2013/408280](https://doi.org/10.1155/2013/408280).
3. Pisani, P. H., & Lorena, A. C. (2013). A systematic review on keystroke dynamics. *Pattern Analysis and Applications*, 16(4), 473–495. [DOI: 10.1007/s13173-013-0117-7](https://doi.org/10.1007/s13173-013-0117-7).
4. Teotia, V., Sharma, S., & Elmiligi, H. (2023). A comprehensive study of machine learning approaches for keystroke dynamics authentication. *International Journal of Biometrics*, 15(2), 148–172. [DOI: 10.1504/IJBM.2023.133148](https://doi.org/10.1504/IJBM.2023.133148).
5. Subash, A., Ghosh, R., Debbarma, K., & Bhowmik, M. K. (2026). Adaptability of current keystroke and mouse behavioral biometric systems: A survey. *Computers & Security*, 150, Article 104731. [DOI: 10.1016/j.cose.2025.104731](https://doi.org/10.1016/j.cose.2025.104731).
