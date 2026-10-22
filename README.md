# Cancer Classification Using Probabilistic Machine Learning

A Probability and Statistics (BAMAT207) course project applying Bayes' theorem, conditional probability, Gaussian modelling, and likelihood estimation to two **separate** public-dataset classification experiments.

> **Scope note:** This is an educational machine-learning project, not a clinically validated diagnostic or screening system. Do not use its predictions for medical decisions.

## Project overview

### 1. Breast Cancer Wisconsin (Diagnostic)

Predicts the dataset's recorded diagnosis label (benign or malignant) from 30 numeric nucleus-measurement features. There are 569 rows in the original dataset. A stratified 80:20 train/test split is used.

### 2. Cervical Cancer Risk Factors

Predicts the dataset's recorded `Biopsy` outcome from selected risk-factor/history features. The cervical data are not merged with the breast data, and other diagnosis/screening outcomes are excluded from the model inputs to reduce target leakage. The positive class is rare, so precision, recall, balanced accuracy, and precision-recall metrics matter more than accuracy alone.

## Methods

- Exploratory data analysis and class distributions
- Gaussian Naive Bayes (GNB) implemented from scratch
- Comparison against scikit-learn GaussianNB
- Standard and class-balanced Logistic Regression
- Prior-only baseline
- Confusion matrices, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and average precision (PR-AUC)
- Stratified five-fold cross-validation on the training partition
- Exported predictions, tables, and diagnostic plots

## Key test-set findings

On the fixed breast test split, scratch Gaussian Naive Bayes achieves **93.86% accuracy**, **83.33% malignant-class recall**, **100% malignant-class precision**, and **0.993 ROC-AUC**. Balanced Logistic Regression reaches **97.37% accuracy** and **95.24% malignant-class recall** on this split.

On the cervical biopsy-outcome test split, Gaussian Naive Bayes has **90.91% positive-class recall but 6.33% precision**, with many false-positive predictions. The cervical results illustrate severe class imbalance and limitations of using Gaussian class-conditional densities on mixed/discrete risk-factor variables.

The from-scratch Gaussian Naive Bayes implementation produces identical hard predictions to scikit-learn GaussianNB on both held-out test sets in the recorded validation run. This checks implementation agreement, not the clinical correctness of either model.

## Repository structure

```text
.
├── Multi_Cancer_Probabilistic_Classification.ipynb
├── data/
│   ├── data.csv
│   ├── risk_factors_cervical_cancer.csv
│   └── SOURCES.md
├── figures/                       # Generated EDA/evaluation plots
├── results/                       # Metrics, CV, validation, predictions (CSV)
├── report/
│   ├── Cancer_Classification_IEEE_Report.tex
│   ├── Cancer_Classification_IEEE_Report.pdf
│   └── figures/                   # Figures referenced by the LaTeX report
├── requirements.txt
└── .gitignore
```

## Run locally (macOS, Linux, or Windows)

1. Install Python 3.10 or newer.
2. Open Terminal in this repository folder.
3. Create and activate an environment (macOS/Linux):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `Multi_Cancer_Probabilistic_Classification.ipynb` and choose **Run All**. The notebook loads CSVs from `data/` and writes new plots to `figures/` and tables/predictions to `results/`.

## Run in Google Colab

Open the notebook in Colab. The setup cell mounts Google Drive and uses `MyDrive/Cancer_Classification_Probabilistic_ML` for persistent inputs and outputs. If either CSV is missing from that Drive folder, the notebook asks you to upload it once and saves it there. Then choose **Runtime → Run all**.

## Report

The IEEE conference-style report is available as a PDF in `report/`. To edit and compile the report, open `report/Cancer_Classification_IEEE_Report.tex` in a TeX environment with the IEEEtran class and required packages. See [`report/README.md`](report/README.md).

## Dataset sources

Source links, attribution, and interpretation notes are listed in [`data/SOURCES.md`](data/SOURCES.md). Please check the dataset hosts' usage and attribution terms before redistributing the raw data elsewhere.

## Reproducibility and limitations

- The notebook uses a fixed random seed and stratified train/test splits.
- All learned imputation and model parameters are fitted on training data only.
- Five-fold CV is run on the training partition, separate from held-out test evaluation.
- Gaussian Naive Bayes assumes conditional independence among predictors; correlated breast measurements and mixed/discrete cervical variables challenge that assumption.
- Results depend on the selected split and preprocessing. They are not external validation and must not be described as clinical performance.
