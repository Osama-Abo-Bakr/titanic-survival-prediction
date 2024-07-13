# Titanic Survival Prediction

## Project Overview

This project aims to predict the survival of Titanic passengers using various machine learning models. The process includes data cleaning, feature engineering, model training, and evaluation to achieve the best possible prediction accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

## Introduction

This project leverages machine learning to predict the survival of passengers on the Titanic. By analyzing the given dataset, we apply various models to determine which features most significantly impact survival rates.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **TensorFlow/Keras**: Neural network implementation
- **XGBoost**: Gradient boosting algorithms

## Data Preprocessing

1. **Data Loading**:
   - Loaded training and test datasets using `pd.read_csv()`.

2. **Data Cleaning**:
   - Dropped irrelevant columns: `Name`, `Cabin`, `Ticket`, `PassengerId`.
   - Filled missing values for `Embarked`, `Age`, and `Fare`.

3. **Outlier Removal**:
   - Removed outliers in `Fare` and `Parch`.

## Feature Engineering

1. **Log Transformation**:
   - Applied log transformation to `SibSp` and `Fare` to reduce skewness.

2. **Label Encoding**:
   - Encoded categorical features `Sex` and `Embarked` using `LabelEncoder`.

3. **Visualization**:
   - Created histograms, count plots, and correlation heatmaps to understand feature relationships.

## Modeling

1. **Logistic Regression**:
   - Implemented a logistic regression model with `max_iter=2000`.

2. **Random Forest Classifier**:
   - Built a Random Forest model with 500 estimators and a maximum depth of 20.

3. **AdaBoost Classifier**:
   - Used AdaBoost with a base decision tree classifier.

4. **XGBoost Classifier**:
   - Applied XGBoost for gradient boosting.

5. **Neural Network**:
   - Designed and trained a neural network using TensorFlow/Keras.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/titanic-survival-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd titanic-survival-prediction
   ```


## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict survival on the test data.

## Conclusion

This project demonstrates the use of various machine learning models to predict Titanic passenger survival. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors influencing survival rates.
---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow.keras as k
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv(r"D:\Pycharm\Titanic_3\train.csv")
test = pd.read_csv(r"D:\Pycharm\Titanic_3\test.csv")
test_ID = test["PassengerId"]

# Data cleaning
data = data.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1)
test = test.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1)

data.Embarked.fillna("U", inplace=True)
test.Embarked.fillna("U", inplace=True)

cols = ["Age", "Fare"]
for col in cols:
    data[col].fillna(data[col].mean(), inplace=True)
    test[col].fillna(data[col].mean(), inplace=True)

data.drop(data[data["Fare"] > 250].index, axis=0, inplace=True)
data.drop(data[data["Parch"] > 5].index, axis=0, inplace=True)

data["SibSp"] = np.log(data["SibSp"]).replace([-np.inf, np.inf], 0)
test["SibSp"] = np.log(test["SibSp"]).replace([-np.inf, np.inf], 0)

data["Fare"] = np.log(data["Fare"]).replace([-np.inf, np.inf], 0)
test["Fare"] = np.log(test["Fare"]).replace([-np.inf, np.inf], 0)

# Visualizations
sns.countplot(x="Survived", hue="Sex", data=data)
sns.countplot(x="Survived", hue="Pclass", data=data)
sns.countplot(x="Survived", hue="Embarked", data=data)

# Label encoding
la = LabelEncoder()
col_s = ["Sex", "Embarked"]
for col in col_s:
    data[col] = la.fit_transform(data[col])
    test[col] = la.fit_transform(test[col])

# Splitting data
X = data.drop(["Survived"], axis=1)
Y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)
print(f"The Training score ==> {model.score(x_train, y_train)}")
print(f"The Testing score ==> {model.score(x_test, y_test)}")
accuracy_score(y_test, model.predict(x_test))

# Random Forest Classifier
model1 = RandomForestClassifier(n_estimators=500, max_depth=20, max_features=2, min_samples_leaf=5)
model1.fit(x_train, y_train)
print(f"The Training score ==> {model1.score(x_train, y_train)}")
print(f"The Testing score ==> {model1.score(x_test, y_test)}")

# AdaBoost Classifier
Adaboost_reg = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, min_samples_leaf=5), n_estimators=200, learning_rate=1)
Adaboost_reg.fit(x_train, y_train)
print(f"The predict Score Train is ==> {Adabo

ost_reg.score(x_train, y_train)}")
print(f"The predict Score Test is ==> {Adaboost_reg.score(x_test, y_test)}")

# XGBoost Classifier
model_xgb = xgb.XGBClassifier(n_estimators=60, max_depth=60000, learning_rate=0.1, min_child_weight=4, random_state=42)
model_xgb.fit(x_train, y_train)
print(f"The predict Score Train is ==> {model_xgb.score(x_train, y_train)}")
print(f"The predict Score Test is ==> {model_xgb.score(x_test, y_test)}")

# Neural Network
from keras.utils import to_categorical
label = to_categorical(Y, 2)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, label, train_size=0.8, random_state=42)

model_nn = k.models.Sequential([
    k.layers.Dense(512, activation="relu"),
    k.layers.Dense(128, activation="relu"),
    k.layers.Dense(256, activation="relu"),
    k.layers.Dense(128, activation="relu"),
    k.layers.Dense(64, activation="relu"),
    k.layers.Dense(32, activation="relu"),
    k.layers.Dense(16, activation="relu"),
    k.layers.Dense(2, activation="softmax")
])

model_nn.compile(optimizer="adam", loss=k.losses.CategoricalFocalCrossentropy(), metrics=["accuracy"])
history = model_nn.fit(x_train1, y_train1, epochs=1000, validation_data=(x_test1, y_test1), validation_split=0.2)
```
