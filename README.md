# House Price Prediction — Linear Regression

##  Project Overview

This project is a hands-on **House Price Prediction** project using **Linear Regression**.

The goal is to learn the complete machine learning workflow on a small, practical dataset — from understanding the data to preprocessing, training a model, evaluating its performance, and analyzing its errors.

This project focuses on **Linear Regression first**, so the workflow and model evaluation can be understood clearly before moving to other algorithms.

##  Dataset

The dataset contains 100 house records with:

- `area` — area of the house
- `bedrooms` — number of bedrooms
- `age` — age of the house
- `location` — location category such as City or Suburb
- `price` — target house price

The dataset contains missing values in `area`, `bedrooms`, `age`, and `location`. The target `price` has no missing values.

##  Exploratory Data Analysis

The project includes:

- Descriptive statistics
- Missing-value analysis
- Correlation
- Scatter plots
- Group-wise price analysis
- Boxplots

Main observations include a strong positive relationship between `area` and `price`, a positive relationship between `bedrooms` and `price`, and a weak relationship between `age` and `price`.

##  Data Preprocessing

The workflow is:

```text
Raw Data
   ↓
Train/Test Split
   ↓
Preprocessing
   ├── Numerical → Median Imputation
   └── Categorical → Most-Frequent Imputation → One-Hot Encoding
   ↓
Linear Regression
   ↓
Predictions
   ↓
Evaluation
```

The train/test split is performed **before fitting preprocessing steps** to avoid data leakage.

### Numerical features

```text
area
bedrooms
age
```

Missing numerical values are handled using **median imputation**.

### Categorical feature

```text
location
```

Missing categorical values are handled using the **most frequent value**, followed by **One-Hot Encoding**.

`ColumnTransformer` is used to apply different preprocessing operations to numerical and categorical features.

##  Model — Linear Regression

The main algorithm is **Linear Regression**.

The model learns the relationship between:

```text
area
bedrooms
age
location
        ↓
     price
```

##  Model Evaluation Metrics

### R² Score

R² measures how much of the variation in the target is explained by the model.

For this project:

```text
Train R² ≈ 0.9733
Test R²  ≈ 0.9654
```

The test R² of approximately **0.9654** means the model explains about **96.54% of the variation in house prices on this test split**.

> R² is not the same as prediction accuracy.

### MSE — Mean Squared Error

MSE calculates the average squared prediction error.

```text
Train MSE ≈ 68.16
Test MSE  ≈ 88.36
```

Because errors are squared, larger errors have a stronger effect.

**Lower MSE is better.**

### MAE — Mean Absolute Error

MAE calculates the average absolute difference between actual and predicted prices.

```text
Test MAE ≈ 6.67
```

It remains in the same units as the target price, making it easy to interpret.

**Lower MAE is better.**

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

```text
Test RMSE ≈ 9.40
```

It is in the original target units while still giving larger errors more influence than MAE.

**Lower RMSE is better.**

##  Train Performance vs Test Performance

The project compares:

```text
Train R² ≈ 0.9733
Test R²  ≈ 0.9654
```

The gap is approximately:

```text
0.0078
```

There is no obvious large train/test performance gap for this split.

Train/test comparison is useful for checking whether the model behaves similarly on data it learned from and unseen data.

##  Residual Analysis

A residual is:

```text
Residual = Actual − Predicted
```

For example:

```text
Actual     = 150
Predicted  = 140
Residual   = +10
```

The model underestimated.

If:

```text
Actual     = 150
Predicted  = 160
Residual   = -10
```

The model overestimated.

The residual plot helps check whether prediction errors show an obvious systematic pattern.

The horizontal line at zero is a **zero-error reference line**, not the regression line.

##  Largest Error Analysis

The project calculates:

```python
absolute_error = abs(residual)
```

Absolute error tells us the **size of the error without considering its direction**.

For example:

```text
+10 → absolute error = 10
-10 → absolute error = 10
```

This helps identify the largest prediction mistakes regardless of whether the model overpredicted or underpredicted.

##  Concepts Practiced

- Data inspection
- Missing-value analysis
- Exploratory Data Analysis
- Correlation
- Train/Test Split
- Data Leakage Prevention
- `SimpleImputer`
- Median Imputation
- Most-Frequent Imputation
- `OneHotEncoder`
- `ColumnTransformer`
- `Pipeline`
- Linear Regression
- Predictions
- R² Score
- MSE
- MAE
- RMSE
- Residual Analysis
- Absolute Error Analysis
- Train vs Test Performance

##  Repository Structure

```text
House-Price-Prediction/
│
├── House_price_prediction_LinearRegression_Notes.ipynb
├── README.md
│
└── house_price_practice.csv
```

##  How to Run

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
House_price_prediction_LinearRegression_Notes.ipynb
```

Run the notebook cells in order.

##  Learning Goal

The purpose of this project is to understand the complete machine learning workflow:

```text
Understand Data
      ↓
EDA
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Feature Transformation
      ↓
Train Model
      ↓
Make Predictions
      ↓
Evaluate Metrics
      ↓
Analyze Errors
```

The project is being used to understand **one machine learning algorithm at a time** before moving to another algorithm.

##  Author

**Rahul**

This project is part of my machine learning learning journey.
