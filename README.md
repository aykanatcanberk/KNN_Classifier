# Breast Cancer Classification using K-Nearest Neighbors (KNN)

This project implements a K-Nearest Neighbors (KNN) classifier to detect breast cancer (benign or malignant) based on features extracted from tumor samples. The pipeline includes data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation using various performance metrics.

---

## Dataset

- **Source:** The dataset used is `data.csv` (Breast Cancer Wisconsin dataset).
- **Samples:** 569
- **Features:** 30 numeric features related to tumor characteristics (radius, texture, smoothness, concavity, symmetry, fractal dimension, etc.) plus a `diagnosis` label.
- **Target Variable:** `diagnosis` (`B` = Benign, `M` = Malignant)

### Preprocessing

1. Removed unnecessary columns:
   - `id` (identifier, not relevant for prediction)
   - `Unnamed: 32` (entirely NaN)
2. Label Encoding:
   - `B` → 0 (Benign)
   - `M` → 1 (Malignant)
3. High correlation features were removed to reduce multicollinearity:
   - Examples: `perimeter_mean`, `area_mean`, `concave points_mean`, `perimeter_worst`, `texture_mean`, `concavity_worst`, etc.
4. Final feature set: 15 features + target variable.

---

## Feature Analysis

- Correlation matrices were generated **before and after feature reduction**.
- Highly correlated pairs (|r| > 0.9) were removed to simplify the model and improve generalization.
- Feature distributions were analyzed using descriptive statistics.

---

## Train-Test Split

- **Training set:** 455 samples
- **Test set:** 114 samples
- **Class distribution:** 357 benign, 212 malignant

---

## KNN Model and Hyperparameter Tuning

- **Hyperparameters tuned via GridSearchCV:**
  - `n_neighbors`: [3, 5, 7, 9, 11, 13, 15]
  - `metric`: ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
  - `weights`: ['uniform', 'distance']

- **Best parameters from cross-validation:**
  - `n_neighbors`: 3
  - `metric`: manhattan
  - `weights`: uniform
  - Best CV Accuracy: 0.9385

- Various KNN configurations were evaluated to compare performance.

---

## Performance Metrics

| Configuration              | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
|----------------------------|----------|-----------|---------------------|------------|----------|---------|
| K=3, Euclidean             | 0.947    | 0.930     | 0.930               | 0.958      | 0.930    | 0.967   |
| K=5, Euclidean             | 0.965    | 0.976     | 0.930               | 0.986      | 0.952    | 0.969   |
| K=7, Euclidean             | 0.947    | 0.974     | 0.884               | 0.986      | 0.927    | 0.978   |
| K=5, Manhattan             | 0.956    | 0.975     | 0.907               | 0.986      | 0.940    | 0.979   |
| K=5, Minkowski             | 0.965    | 0.976     | 0.930               | 0.986      | 0.952    | 0.969   |
| K=5, Chebyshev             | 0.930    | 0.949     | 0.860               | 0.972      | 0.902    | 0.977   |
| K=5, Euclidean, Weighted   | 0.965    | 0.976     | 0.930               | 0.986      | 0.952    | 0.969   |
| **Best Model**             | 0.947    | 0.951     | 0.907               | 0.972      | 0.929    | 0.962   |

- **Best F1-Score:** K=5, Euclidean → F1 = 0.952  
- **Highest Sensitivity:** K=3, Euclidean → 0.930 (only 3 patients missed)

---

## Visualizations

1. **Correlation Heatmaps:** Before and after feature reduction.
2. **Confusion Matrices:** For each KNN configuration.
3. **ROC Curves:** Displayed all configurations with AUC scores.
4. **Performance Comparison:** Bar charts comparing Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC.

---

## Analysis and Recommendations

- **K Value Effect:** Small K can lead to overfitting; large K can underfit.  
- **Distance Metric Effect:** Different metrics perform differently based on data distribution.  
- **Medical Significance:** Sensitivity (Recall) is critical — minimizing false negatives is crucial in cancer detection.  
- **Best Model Recommendation:** K=5, Euclidean metric, uniform weights. This configuration balances overall accuracy and sensitivity for reliable tumor classification.

---

## Requirements

- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

---

## Usage

```python
# Load dataset
dataset = pd.read_csv('data.csv')

# Preprocess, train, and evaluate KNN using the provided notebook script
# View results and visualizations
