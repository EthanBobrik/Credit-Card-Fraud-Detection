# 📊 Credit Card Fraud Detection

## Overview

This project aims to detect fraudulent credit card transactions using supervised machine learning techniques. Due to the highly imbalanced nature of the dataset, special attention was given to evaluation metrics that better reflect performance on the minority (fraud) class, such as **Precision-Recall AUC (PR-AUC)**.

---

## 🧠 Models Used

Several models were trained and evaluated, including:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Neural Network (Keras)**

The primary goal was to **maximize recall** on fraudulent cases while minimizing false positives, as missing a fraud is costlier than investigating a false alarm.

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Handled missing values (if any)
   - Applied scaling via `MinMaxScaler` or `RobustScaler`
   - Split data into stratified training, validation, and test sets

2. **Class Imbalance Handling**
   - Evaluated techniques like **RandomUnderSampling** and **Feature Engineering**
   - Tuned class weights and thresholds

3. **Model Evaluation**
   - Metrics: **PR-AUC**, **ROC-AUC**, **Precision**, **Recall**, **F1-Score**
   - PR-AUC chosen as the most informative metric for imbalanced data

4. **Cross-Validation**
   - Used Stratified K-Fold Cross Validation for consistent performance estimation

5. **Hyperparameter Tuning**
   - Grid search or halving grid search for scikit-learn models
   - Manual and callback-based tuning for Keras models

---

## 📈 Key Results

| Model           | Resampling            | CV PR-AUC | Test PR-AUC | Best Params |
|----------------|------------------------|------------|--------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Logistic**    | None                   | 0.730      | 0.819        | `{'C': 0.359, 'penalty': 'l1'}` |
| **Logistic**    | RandomUnderSampler     | 0.660      | 0.676        | `{'clf__C': 0.0001, 'clf__penalty': 'l2'}` |
| **RandomForest**| None                   | 0.844      | 0.895        | `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 300}` |
| **RandomForest**| RandomUnderSampler     | 0.652      | 0.738        | `{'clf__max_depth': 2, 'clf__max_features': 'sqrt', 'clf__min_samples_split': 2, 'clf__n_estimators': 300}` |
| **XGBoost**     | None                   | **0.851**  | **0.902**    | `{'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}` |
| **XGBoost**     | RandomUnderSampler     | 0.673      | 0.801        | `{'clf__colsample_bytree': 0.6, 'clf__learning_rate': 0.01, 'clf__max_depth': 5, 'clf__n_estimators': 200, 'clf__subsample': 0.6}` |
| **Keras NN**    | None                   | 0.7165     | 0.819        | |
| **Keras NN**    | RandomUnderSampler     | 0.4748     | 0.569        | |
| **Keras NN**    | RandomUnderSampler & L2 Regalurization    | 0.4922     | 0.005        | |


> ✅ **Conclusion**: XGBoost achieved the best tradeoff between high recall and moderate precision, making it the most effective model for detecting fraud.

---

## 🔍 Findings

- **High recall and low precision** is expected and acceptable for fraud detection use cases.
- **PR-AUC > ROC-AUC** is more informative when classes are imbalanced.
- Model generalization was confirmed by test PR-AUC > cross-validation PR-AUC.
- Feature importance from tree-based models helped in interpreting predictions.

---

## ✅ Conclusion

- **XGBoost** was selected as the final model for its strong performance on both recall and PR-AUC.
- Future enhancements may include:
  - Real-time deployment via API
  - Advanced anomaly detection
  - Threshold optimization for precision-recall balance

---

## 📁 How to Use

1. Clone this repository
2. Install dependencies via `requirements.txt`
3. Run the notebook `CreditCardFraudDetection.ipynb`

### Load Saved Models

```python
# Load Pickle model
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Keras model
from tensorflow.keras.models import load_model
model = load_model('model.h5')
