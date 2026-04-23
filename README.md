# 📘 MINI_PROJECT — RVFL-Based Financial Time Series Forecasting

## 📌 Overview

This project implements and evaluates a family of **Random Vector Functional Link (RVFL)** models for **financial time series forecasting**.

The work goes beyond replication and focuses on:
- Cross-architecture comparison  
- Hyperparameter sensitivity analysis  
- Aggregation strategy evaluation (mean vs median)  
- Baseline benchmarking (including Ridge regression)  
- Computational tradeoff analysis  

---

## 🎯 Objectives

- Reproduce and validate RVFL-based forecasting approaches  
- Evaluate the necessity of:
  - Deep architectures (edRVFL)  
  - Recurrent architectures (RedRVFL)  
  - Signal decomposition (EWT)  
- Compare against classical and ML baselines  
- Analyze diminishing returns in model complexity  

---

## 🧠 Models Implemented

### 🔹 RVFL Family

| Model | Description |
|------|------------|
| **RVFL** | Random feature mapping + ridge regression |
| **edRVFL** | Ensemble Deep RVFL (stacked layers) |
| **EWTedRVFL** | RVFL + Empirical Wavelet Transform |
| **RedRVFL** | Recurrent (LSTM-based) extension of edRVFL |

---

### 🔹 Baseline Models

- ARIMA  
- Persistence model  
- SVR  
- TCN  
- LSTM  
- Ridge Regression  

---

## 📂 Project Structure

```

MINI_PROJECT/
│
├── RVFL_Datasets/                # Financial datasets
│
├── ablation_plots/              # Visualization outputs
├── hyperparameter_isolates/     # Sensitivity experiments
├── mean-aggregation/            # Mean aggregation results
├── median-aggregation/          # Median aggregation results
│
├── docs/                        # Notes / documentation
│
├── src/
│   ├── run/
│   │   ├── config.py            # Hyperparameter configurations
│   │   ├── data_loader.py       # Data loading + windowing
│   │   ├── evaluator.py         # Metrics + inverse scaling
│   │   ├── model_runner.py      # Model training + prediction
│   │   ├── run_experiment.py    # Hyperparameter search loop
│   │
│   ├── architecture.py          # Model implementations
│   ├── hyperparameters.py       # Search space definitions
│   ├── metrics.py               # RMSE, MAE, MAPE
│   ├── red_revfl_orchestrator.py# RedRVFL pipeline
│   ├── ridge_baseline.py        # Ridge baseline implementation
│   ├── final_evaluation.py      # Final test evaluation
│   ├── analyze_results.py       # Result aggregation
│   ├── visualize.py             # Plotting utilities
│
├── venv/
├── .gitignore

````

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/anush2006/red-rvfl
cd MINI_PROJECT
````

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### 🔹 1. Hyperparameter Tuning (70/10/20 split)

```bash
python src/run/run_experiment.py
```

* Runs configuration search
* Outputs RMSE, MAE, MAPE, and training time

---

### 🔹 2. Aggregation Experiments

* Mean aggregation → `mean-aggregation/`
* Median aggregation → `median-aggregation/`

Used to compare ensemble behavior.

---

### 🔹 3. Final Evaluation (80/20 split)

```bash
python src/final_evaluation.py
```

* Uses top configurations
* Trains on (train + validation)
* Evaluates on test set

---

### 🔹 4. Ridge Baseline

```bash
python src/ridge_baseline.py
```

* Linear autoregressive baseline
* Used to quantify benefit of nonlinear feature mapping

---

### 🔹 5. Analyze Results

```bash
python src/analyze_results.py
```

* Extract best configurations
* Generate comparison tables

---

### 🔹 6. Visualization

```bash
python src/visualize.py
```

* Heatmaps
* Performance plots
* Ablation visualizations

---

## 📊 Evaluation Metrics

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **MAPE** — Mean Absolute Percentage Error

---

## 🧪 Experimental Design

### Data Processing

* Sliding window transformation
* Min-max scaling (train-based normalization)
* Sequential splitting (no shuffling)

---

### Dataset Splits

| Stage                 | Split           |
| --------------------- | --------------- |
| Hyperparameter tuning | 70% / 10% / 20% |
| Final evaluation      | 80% / 20%       |

---

## 📈 Key Findings

### 1. Linear vs Nonlinear Models

* Ridge ≈ ARIMA → poor performance
* RVFL significantly outperforms linear baselines

**Conclusion:**
Nonlinear feature transformation is essential.

---

### 2. RVFL vs Advanced Architectures

* RVFL provides major gains
* edRVFL / RedRVFL provide marginal improvements

**Conclusion:**
Diminishing returns beyond base RVFL.

---

### 3. Hyperparameter Sensitivity

* Minimal variation across configurations

**Conclusion:**
Model operates in a flat performance region.

---

### 4. Aggregation Strategy

* Mean ≈ Median

**Conclusion:**
Low ensemble diversity → aggregation has negligible impact.

---

### 5. Computational Tradeoff

* Training time varies significantly
* Performance remains stable

**Conclusion:**
Higher complexity does not yield proportional benefit.

---

## 🧠 Core Insight

> The primary performance gains come from nonlinear feature transformation, while additional architectural complexity provides only marginal improvements.

---

## 📊 Hierarchy of Impact

| Factor                        | Impact     |
| ----------------------------- | ---------- |
| Nonlinear feature mapping     | HIGH       |
| Model class (RVFL vs linear)  | HIGH       |
| Architecture (deep/recurrent) | LOW        |
| Hyperparameters               | VERY LOW   |
| Aggregation method            | NEGLIGIBLE |

---

## ⚠️ Reproducibility Notes

Results may differ from the original paper due to:

* Different window sizes
* Scaling variations
* Limited hyperparameter search
* Implementation differences

Focus is on:

> **trend consistency, not exact numerical replication**

---

## 📌 Limitations

* Low ensemble diversity
* Random feature mapping is not optimized
* No statistical significance testing
* Computational inefficiency at higher complexity

---

## 🔮 Future Work

* Learnable feature mappings
* Improve ensemble diversity
* Statistical validation (e.g., Wilcoxon test)
* Optimize computational efficiency
* Explore hybrid architectures

---

## 🧾 License

Academic / research use.

---


## ✅ Summary

This project delivers:

* End-to-end RVFL implementation
* Fair baseline comparisons (including Ridge)
* Ablation and sensitivity analysis
* Critical evaluation of architectural complexity

**Key takeaway:**

> Most performance gains arise from nonlinear feature transformation, not model complexity.

```

