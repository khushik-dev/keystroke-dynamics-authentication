# CSL236 Viva Defense & Technical Q&A Guide

**Project:** Machine Learning-Based User Authentication Using Keystroke Dynamics  
**Course:** CSL236 – Introduction to Artificial Intelligence & Machine Learning  
**Target Student Audience:** Oral Viva Defense Preparation (B.Tech CSE Semester 5)  

---

## 1. Core Technical Defense Questions

### Q1: Why Random Forest Was Selected for Production (Why choose Random Forest when another model performed better?)

| Model Configuration | Algorithm Family | EER | ROC-AUC | FAR | FRR |
|---|---|---:|---:|---:|---:|
| Logistic Regression (Baseline) | Linear Model | 17.00% | 0.9017 | **0.15%** | 99.50% |
| Balanced Logistic Regression | Linear Model | **14.44%** | **0.9161** | 9.91% | **25.00%** |
| Random Forest *(Serving)* | Tree Ensemble | 20.03% | 0.8582 | 12.05% | 54.50% |
| Support Vector Machine (RBF) | Kernel Method | 22.91% | 0.8223 | **4.88%** | 33.00% |

* **How to answer verbally:**  
  "We evaluated four configurations across **three algorithm families**: Linear Models (unweighted and balanced Logistic Regression), Decision Tree Ensembles (Random Forest), and Kernel Methods (RBF SVM).
  * **Empirical Reality:** Balanced Logistic Regression achieved our lowest Equal Error Rate (14.44%) and highest ROC-AUC (0.9161), while SVM achieved our lowest False Acceptance Rate (4.88%). Random Forest was **not** our most accurate model on raw biometric error rates.
  * **Why We Retained Random Forest:** We retained Random Forest because of an engineering and explainability trade-off. Random Forest natively provides tree-based Mean Decrease in Impurity (MDI) feature importances (`classifier.feature_importances_`). In our web application, this powers real-time diagnostic tips that show users exactly which keystroke timings caused an authentication failure. Linear and kernel models require expensive post-hoc explainers like SHAP, which add latency to real-time serving. It also matches the Brainmentors reference architecture (`mlops-starter`).
  * **Holdout Integrity:** We did not use test sessions 5–8 to pick or tune Random Forest. When test results showed Balanced Logistic Regression performed better, we chose to be transparent about the trade-off and retain Random Forest for its UI explainability, rather than retroactively switching models based on holdout scores."

---

### Q2: What happens if an input contains NaN values or out-of-range numbers?
* **How to answer verbally:**  
  "Every incoming prediction request to `POST /predict` is intercepted and validated by a strict **Pydantic schema (`KeystrokeInput`)** in `app.py` before it ever reaches the Scikit-learn model:
  1. If any of the 31 features is missing, `null`, `NaN`, or an invalid type, Pydantic immediately rejects the request and returns an **HTTP 422 Unprocessable Entity** error showing exactly which field failed.
  2. If a value falls outside our physical data contract bounds (for example, a hold time greater than 3 seconds or a flight time greater than 30 seconds), it also returns an HTTP 422 error.
  3. However, we explicitly allow Keyup-to-Keydown ($UD$) features to be negative down to $-1.0$s. That is because in fluent typing, people often press the next key before releasing the previous one (key rollover). In our dataset, 10.84% of all $UD$ entries are negative, and preserving them is essential for authentic biometrics."

---

### Q3: Why do you have two requirements files (`requirements.txt` vs `requirements-serve.txt`)?
* **How to answer verbally:**  
  "This is a standard MLOps best practice to keep our production container lean and secure:
  1. `requirements.txt` is our **full development and training environment**. It includes heavy libraries like MLflow, Pytest, Matplotlib, and Scipy, which we need for training models, plotting ROC curves, and running test suites.
  2. `requirements-serve.txt` is our **lean serving runtime**. In production, we only need to load the serialized pipeline and serve predictions using FastAPI and Uvicorn. By excluding MLflow and training dependencies, we keep our Docker image lightweight, speed up container build times, reduce memory usage, and shrink the attack surface."

---

### Q4: How would you know if the model is degrading in production? (Data Drift)
* **How to answer verbally:**  
  "We built an automated data drift monitoring script (`monitoring/drift_check.py`) and an asynchronous prediction logger:
  1. Every prediction request is logged to `monitoring/prediction_logs.jsonl`. For privacy reasons, we log only the 31 timing numbers and the resulting match score—we never log passwords, usernames, or IP addresses.
  2. The drift detection script compares batches of production inputs against our enrollment baseline (Sessions 1–4) across all 31 features using the **two-sample Kolmogorov-Smirnov (KS) test** and percentage mean shift.
  3. If more than 5 features show statistically significant shift ($p < 0.05$ and mean shift $> 15\%$), the script flags the status as `DRIFT_DETECTED` in `monitoring/drift_report.json`. This signals that the user's typing pattern has shifted (due to habituation, a new keyboard, or fatigue), alerting us to re-enroll or recalibrate."

---

### Q5: Why was your midterm baseline achieving 97.90% accuracy but a 99% False Rejection Rate?
* **How to answer verbally:**  
  "This is a classic example of the **Accuracy Paradox under extreme class imbalance**. 
  In our 1-vs-all setup with 51 subjects, genuine attempts (`s002`) make up only 200 out of 10,200 test samples—about 1.96%. Impostors make up 98.04%.
  
  A default classifier optimizing standard accuracy with a 0.5 decision threshold discovers that if it simply predicts 'impostor' for every single attempt, it achieves $10,000 / 10,200 \approx 98.04\%$ accuracy! 
  
  However, out of 200 genuine logins, it accepted only 1 attempt and rejected 199. That produced a **99.50% False Rejection Rate**, making the system completely unusable for the real user.
  
  We resolved this in the end-term by:
  1. Adding `class_weight='balanced'` so the loss function penalizes mistakes on genuine samples 50 times more heavily.
  2. Sweeping the operating threshold on Session 4 validation data to find the **Equal Error Rate (EER)** point ($\tau = 0.4445$), which brought genuine recall up from 0.5% to **75.00%**."

---

### Q6: Why did you use a session-based split instead of a random 80/20 train/test split?
* **How to answer verbally:**  
  "In the CMU benchmark, each user typed 50 repetitions per session across 8 distinct sessions recorded on different days.
  
  If we used a random shuffle split like `train_test_split(shuffle=True)`, repetitions recorded just seconds apart in the same 15-minute sitting would be split between train and test sets. The model would simply memorize that day's temporary typing rhythm and fatigue quirks, causing **data leakage** and unrealistically high test scores.
  
  Instead, we enforced a strict temporal split:
  * **Sessions 1–3** for training.
  * **Session 4** for validation and EER threshold tuning.
  * **Sessions 5–8** as an untouched holdout test set to measure genuine performance on future days.
  For our production artifact, we serialize the exact model fitted on Sessions 1–3 paired directly with the Session 4 validation threshold ($\tau = 0.1094$). We deliberately chose not to refit on Sessions 1–4 to maintain a 100% paired calibration without assuming the threshold transfers to a changed tree ensemble."

---

### Q7: Explain what a negative $UD$ value physically means.
* **How to answer verbally:**  
  "$UD$ stands for Keyup-to-Keydown. Mathematically:
  $$UD(k_1, k_2) = t_{\text{press}}(k_2) - t_{\text{release}}(k_1)$$
  
  When someone types fast, they don't wait to lift their finger off key 1 before pressing key 2. Instead, they press key 2 *while* key 1 is still depressed. This is called **key rollover**.
  
  Because key 2 is pressed before key 1 is released ($t_{\text{press}}(k_2) < t_{\text{release}}(k_1)$), the difference is a negative number.
  
  In our dataset, **22,118 transitions (10.84% of all $UD$ entries)** are negative. For instance, `UD.i.e` has 4,811 negative values. If an ML engineer mistakenly clips these to zero, they destroy a critical biometric signal that distinguishes fast, natural typists."

---

### Q8: Why did you use PCA, and how many components did it keep?
* **How to answer verbally:**  
  "We used PCA because keystroke features suffer from severe **multicollinearity**. By definition:
  $$DD(k_1, k_2) \approx H(k_1) + UD(k_1, k_2)$$
  The flight time between two keypresses is physically tied to the hold time of the first key plus the release-to-press interval. In our correlation heatmap, adjacent digraph pairs had correlation coefficients exceeding $0.85$.
  
  By applying `PCA(n_components=0.95)`, we compressed the 31 features into **17 orthogonal principal components**. This eliminated collinearity and noise while retaining $95.2\%$ of the total timing variance."

---

### Q9: What is the Equal Error Rate (EER), and why is it important in biometrics?
* **How to answer verbally:**  
  "The Equal Error Rate is the point on the ROC curve where the **False Acceptance Rate (FAR)** equals the **False Rejection Rate (FRR)**.
  * If the threshold is set too high, the system is very strict: FAR is low, but FRR is high (annoying the genuine user).
  * If the threshold is set too low, the system is too lenient: FRR is low, but FAR is high (letting in impostors).
  
  The EER represents the intrinsic accuracy of a biometric system when security and usability are perfectly balanced. In our testing, Balanced Logistic Regression achieved the best EER of **14.44%**, compared to 20.03% for Random Forest."

---

### Q10: How did you ensure there was no data leakage during preprocessing?
* **How to answer verbally:**  
  "We ensured strict separation between fit and transform operations:
  1. `StandardScaler` and `PCA` were fitted **strictly on the training partition** (Sessions 1–3). The serialized pipeline uses this exact scaler fit.
  2. The validation set (Session 4) and test set (Sessions 5–8) were only passed through `transform()`.
  3. No test set statistics (means, standard deviations, or principal axes) were ever seen during preprocessing or model training."

---

### Q11: Why is your serialized production model trained on Sessions 1–3 rather than refitted on Sessions 1–4?
* **How to answer verbally:**  
  "This was a deliberate methodological decision regarding **calibration consistency**:
  1. When we calibrated our operating threshold ($\tau = 0.1094$) on Session 4 validation data, that threshold was mathematically tied to the probability outputs of the model trained on Sessions 1–3.
  2. If we subsequently refit the Random Forest on Sessions 1–4, the model gains 33% more genuine training repetitions (200 vs 150), which alters the tree splits and predicted probability distributions.
  3. Because Session 4 is now inside the training set, evaluating the refitted model on Session 4 produces heavily biased in-sample scores. We would have no way to verify that $\tau = 0.1094$ remains the true Equal Error Rate threshold without touching holdout Sessions 5–8, which would violate our evaluation integrity.
  4. We chose the **strictly paired approach (Option B)**: the model serving live predictions is the exact model calibrated on Session 4. The trade-off is utilizing 150 genuine training repetitions instead of 200, which avoids unverified threshold extrapolation."

---

## 2. Quick Rubric Mapping Summary

| Rubric Category | Marks | Implementation Evidence |
|---|---|---|
| **Problem & Data Contract** | 10 | 31-feature table in `README.md` with physical bounds enforced via Pydantic in `app.py`. |
| **Modeling & MLflow Tracking** | 15 | `train.py` tracking to `sqlite:///mlflow.db`; 4 model families logged with EER calibration. |
| **Model Serialization** | Stage 3 | `KeystrokeAuthPipeline` wrapper saved to `model/model.joblib`; loaded on app startup. |
| **FastAPI & UI** | 10 | `/health`, `/predict`, `/`, responsive UI in `static/index.html` with diagnostic feedback. |
| **Automated Tests** | 10 | 6 Pytest tests in `tests/test_app.py` passing cleanly (`6 passed in 8.52s`). |
| **Docker (Lean Image)** | 10 | `Dockerfile` using `python:3.11-slim` and `requirements-serve.txt`. |
| **Continuous Integration** | 10 | `.github/workflows/ci.yml` running install, baseline training, pytest, and drift check. |
| **Cloud Deployment & Drift** | 5 | Ready for cloud deployment; `monitoring/drift_check.py` with KS statistical tests. |
| **Report & README** | 5 | Formal report in `reports/Project_Report.md` and complete setup guide in `README.md`. |
| **Presentation (PPT)** | 10 | 10 interactive slides in `presentation/slides.html` with speaker notes. |
| **Viva Defense** | 15 | This viva guide + understanding of biometric error rates and domain physics. |
| **Bonus Criteria** | +5 | UI feature importance (Bonus A) + KS-test drift check (Bonus B) + CI pipeline (Bonus C). |
