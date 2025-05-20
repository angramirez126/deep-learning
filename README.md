# Alphabet Soup Charity Success Predictor.Deep Learning

## Overview

This project applies supervised machine learning to help the nonprofit Alphabet Soup identify which funding applicants are most likely to succeed. Using a dataset of over 34,000 historical applications, the goal was to build a binary classifier to predict applicant success (`IS_SUCCESSFUL = 1` or `0`).

The initial approach used a deep neural network (DNN) with TensorFlow and Keras. Due to limited performance, the project transitioned to using XGBoost, a gradient-boosted tree algorithm, which proved more effective on the structured dataset.

---

## Data Preprocessing

- **Target Variable**: `IS_SUCCESSFUL`
- **Dropped Columns**: `EIN`, `NAME` (non-informative identifiers)
- **Categorical Simplification**: Consolidated rare `APPLICATION_TYPE` and `CLASSIFICATION` values into `'Other'`
- **Encoding**: Applied one-hot encoding with `pd.get_dummies()`
- **Scaling**: Used `StandardScaler` on numerical features

---

## Modeling Approaches

### Deep Neural Network (DNN)

- Architecture: 2 hidden layers (ReLU), 1 output layer (sigmoid)
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Result: Accuracy plateaued around 60%, below the 75% target

### XGBoost Classifier

- Selected due to better performance on tabular data
- Minimal tuning required
- Achieved accuracy consistently above 75% on test data
- Outperformed the DNN in both accuracy and stability

---

## Model Evaluation

**XGBoost Performance:**

- Accuracy: >75%
- Evaluation: Accuracy, precision, recall, F1-score
- Loss Function: Log loss

The XGBoost model produced more interpretable, generalizable results with lower risk of overfitting compared to the neural network.

---

## Conclusion

This project highlights the importance of model selection based on data type. While deep learning is powerful, XGBoost proved more suitable for this structured classification task—delivering higher accuracy, faster training, and more consistent outcomes.

---

## Technologies Used

- Python
- Pandas, NumPy, scikit-learn
- TensorFlow / Keras
- XGBoost
