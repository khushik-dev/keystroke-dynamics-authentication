# REQUIREMENTS CHECKLIST & OFFICIAL RUBRIC MAPPING

**Course:** CSL236 – Introduction to AI & Machine Learning, B.Tech CSE, Sem 5, The NorthCap University  
**Project:** Machine Learning-Based User Authentication Using Keystroke Dynamics  
**Official Weightage:** 100 Marks (Core) + 5 Marks (Bonus)  
**Authoritative Midterm Starting Point:** `reference/Project_Aiml.ipynb` (Enrolled Subject: `s002`, Sessions 1–4 Train vs 5–8 Test)

---

## 1. End-Term Evaluation Rubric Mapping (100 + 5 Marks)

| Stage / Component | Marks | Official Requirement & Criteria | Implementation File / Evidence | Implementation & Verification Status |
|---|---|---|---|---|
| **1. Problem, Dataset & Data Contract** | **10** | Clear problem statement, cited public dataset (CMU DSL-StrongPasswordData), complete data contract table in README with 31 features (type, allowed ranges, physical domain justification) | `README.md`, `data/DSL-StrongPasswordData.csv`, Pydantic schema in `app.py` | **Implemented & Verified** (31 timing features documented; physical key rollover [-1.0, 30.0] preserved) |
| **2. Modelling & MLflow Tracking** | **15** | At least 3 tracked runs with hyperparameters as CLI arguments; at least 2 algorithms compared (Logistic Regression baseline vs RF / SVM); SQLite backend (`sqlite:///mlflow.db`); justified model choice; comprehensive metrics: Acc, Precision, Recall, F1, FAR, FRR, ROC-AUC, EER | `train.py`, `mlflow.db` (5 tracked runs), `README.md` results table | **Implemented & Verified** (4 model families tracked; temporal 3-way split: S1–3 train, S4 val, S5–8 test) |
| **3. Model Serialization** | *(Included in Stage 2/4)* | Save best model to `model/model.joblib`; loaded once at application startup outside request handler | `model/model.joblib`, `app.py` startup load | **Implemented & Verified** (Standalone `KeystrokeAuthPipeline` loaded on startup outside request handler) |
| **4. FastAPI Service, Validation & UI** | **10** | `GET /health`, `POST /predict`, `GET /` (UI). Pydantic schema enforcing contract ranges (invalid returns HTTP 422). Response returns prediction, score, and 2–3 actionable rule-based tips | `app.py`, `static/index.html`, `/docs` Swagger | **Implemented & Verified** (Tested via curl, Swagger UI, and automated test suite) |
| **5. Automated Tests** | **10** | At least 4 pytest tests passing: health check, UI loading, domain-logic test (genuine s002 sample scores higher than impostor sample), boundary-value test expecting HTTP 422 | `tests/test_app.py` | **Implemented & Verified** (6/6 tests passing in 4.50s) |
| **6. Docker (Lean Image)** | **10** | Dockerfile based on `python:3.11-slim`, built using `requirements-serve.txt` only (lean runtime, no dev/training bloat). Builds and runs serving on port 8000; image size documented | `Dockerfile`, `requirements-serve.txt`, `README.md` | **Configuration Complete** (Dockerfile written and validated; local Docker daemon inactive during audit) |
| **7. Continuous Integration (CI)** | **10** | GitHub Actions (`.github/workflows/ci.yml`) on every push to main: installs dependencies, trains baseline model, executes pytest suite; green badge/check | `.github/workflows/ci.yml`, `.gitignore` | **Configuration Complete** (Workflow YAML syntax validated; ready for GitHub push) |
| **8. Cloud Deployment & Monitoring** | **5** | Deploy container with restart policy; log every prediction input to a file for drift monitoring; public working URL on demo day | `monitoring/prediction_logs.jsonl`, `monitoring/drift_check.py` | **Monitoring Implemented; Deployment Pending** (Logging & drift check verified; cloud deployment instructions documented) |
| **9. Report and README** | **5** | Professional `README.md` (complete setup, contract, MLflow results table, screenshots). 6–10 page PDF report (`reports/Project_Report.md`) covering 15 mandatory sections | `README.md`, `reports/Project_Report.md` | **Implemented & Verified** (15-section academic report and comprehensive README created) |
| **10. Presentation (PPT)** | **10** | 8–12 slide presentation (`presentation/slides.html` - exactly 10 slides) with structured flow, diagrams, MLflow comparison, screenshots, clickable links slide, and `presentation/viva_preparation.md` | `presentation/slides.html`, `presentation/viva_preparation.md` | **Implemented & Verified** (10 interactive slides and viva defense preparation document ready) |
| **11. Demo & Viva (Individual)** | **15** | 5-min live demo (valid prediction, 422 rejection, MLflow UI, green CI check) + 5-min individual defense explaining every line of code, math, and tradeoffs | `presentation/viva_preparation.md`, live demo presets | **Prepared for Evaluation** (Demo presets configured in UI; viva defense questions comprehensively prepared) |
| **BONUS 1: Data Drift Detection** | **+2** | Drift check script comparing prediction input distributions (mean/std/KS-test) vs training reference | `monitoring/drift_check.py`, `monitoring/drift_report.json` | **Implemented & Verified** (KS-test and mean shift comparison against baseline) |
| **BONUS 2: Explainability / Feature Importance** | **+2** | UI feature importance or SHAP explanations for predictions | Integrated in `app.py` & `static/index.html` | **Implemented & Verified** (Top-3 timing discrepancies displayed dynamically in UI) |
| **BONUS 3: Automated CD** | **+1** | CD step that triggers automated deployment upon successful CI | Documented in `.github/workflows/ci.yml` & `README.md` | **Documented in CI/CD Guide** |

---

## 2. Objective Status Summary

* **Core ML & Software Artifacts:** Fully implemented, verified, and passing all unit/integration tests.
* **Documentation & Academic Deliverables:** Complete, referenced to 5 peer-reviewed publications, and aligned with official department guidelines.
* **Pending Student Actions for Submission:**
  1. Initialize remote repository on GitHub and push code (`git remote add origin ...`, `git push -u origin main`).
  2. Start Docker Desktop if local container demo is desired (`docker build -t keystroke-auth .`).
  3. (Optional for Bonus) Connect GitHub repository to Render/Railway for live cloud URL.
