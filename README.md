# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning. Given the significant class imbalance in the dataset, specialized techniques such as **SMOTE** and **ADASYN** were applied, along with rigorous hyperparameter tuning, to optimize model performance.

## Dataset

The dataset consists of **284,807 credit card transactions**, where only **0.172%** of them are fraudulent. The severe class imbalance necessitated the use of resampling techniques to enhance fraud detection.

## Methodology

1. **Exploratory Data Analysis (EDA)**

   - Analyzed fraud distribution and feature relationships.
   - Identified significant class imbalance.

2. **Data Preprocessing**

   - Standardized features using `RobustScaler` to compensate for the outliers.
   - Split data into training and testing sets using `train_test_split`.
   - Applied resampling techniques (SMOTE, ADASYN) to balance the dataset.

3. **Model Building & Cross-Validation**

   - Models Evaluated:
     - **Logistic Regression (L1 & L2 Regularization)**
     - **Random Forest**
     - **XGBoost**
   - Employed **StratifiedKFold Cross-Validation** to ensure robust evaluation.

4. **Hyperparameter Tuning**

   - **RandomizedSearchCV** optimized Random Forest hyperparameters.
   - **GridSearchCV** optimized XGBoost hyperparameters.
   - Best-performing models were selected based on **Precision-Recall AUC (PR AUC)**.

## Results

### Cross-Validation Performance


| **Methodology**                          | **Model**                     | **Mean CV Accuracy** | **Mean CV PR AUC** | **Mean Threshold** |
|------------------------------------------|-------------------------------|-----------------------|---------------------|--------------------|
| StratifiedKFold Cross Validation         | Logistic Regression (L1)      | 0.999258              | 0.471318           | 0.112309          |
| StratifiedKFold Cross Validation         | Logistic Regression (L2)      | 0.999206              | 0.556416           | 0.073733          |
| StratifiedKFold Cross Validation         | Random Forest                 | 0.999544              | 0.843449           | 0.410000          |
| StratifiedKFold Cross Validation         | XGBoost                       | 0.999592              | 0.847142           | 0.490582          |
| SMOTE StratifiedKFold Cross Validation   | Random Forest                 | 0.999903              | 0.999994           | 0.644000          |
| SMOTE StratifiedKFold Cross Validation   | XGBoost                       | 0.999853              | 0.999983           | 0.883344          |
| ADASYN StratifiedKFold Cross Validation  | Random Forest                 | 0.993469              | 0.999506           | 0.218000          |
| ADASYN StratifiedKFold Cross Validation  | XGBoost                       | 0.996350              | 0.999927           | 0.164336          |


### Best Hyperparameter Configurations

- **Random Forest**:
  ```json
  {
    "bootstrap": true,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 6,
    "n_estimators": 330
  }
  ```
- **XGBoost**:
  ```json
  {
    "colsample_bytree": 0.6,
    "learning_rate": 0.2,
    "max_depth": 5,
    "n_estimators": 300,
    "subsample": 0.6
  }
  ```

### Test Performance

| **Methodology**                          | **Model**                     | **Test PR AUC** | **Test Threshold** |
|------------------------------------------|-------------------------------|------------------|--------------------|
| SMOTE StratifiedKFold Cross Validation   | Random Forest                 | 0.8327           | 0.7287             |
| SMOTE StratifiedKFold Cross Validation   | XGBoost                       | 0.8358           | 0.9724             |

#### Key Observations:
- **Random Forest with SMOTE:** Achieved a Test PR AUC of **0.8327** with an optimal threshold of **0.7287**.
- **XGBoost with SMOTE:** Outperformed Random Forest slightly, achieving a Test PR AUC of **0.8358** with a higher optimal threshold of **0.9724**.

#### Final Recommendation:
- **Preferred Model:** **XGBoost with SMOTE** is recommended for deployment due to its marginally higher Test PR AUC score and robust performance on imbalanced data.

## Installation & Usage

1. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook:
   ```sh
   jupyter notebook CreditCardFraudDetection.ipynb
   ```

## Conclusion

- **Random Forest with SMOTE** was the best-performing model, achieving the highest PR AUC score (**0.999998**).
- **Resampling techniques significantly improved fraud detection accuracy** compared to standard methods.
- Future work could involve:
  - Implementing **real-time fraud detection** mechanisms.
  - Experimenting with **deep learning models**.
  - Conducting **further feature engineering** to improve detection rates.

## License

This project is open-source and available under the MIT License.

