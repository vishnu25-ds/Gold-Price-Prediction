# Gold Price Prediction and Forecasting

This project uses a gold price dataset to predict the "EUR/USD" exchange rate using various regression models. It involves data cleaning, feature engineering, model training, evaluation, and forecasting for future years. The dataset is preprocessed, outliers are handled, and several machine learning algorithms are applied to make predictions. The models evaluated include Linear Regression, Support Vector Regression (SVR), Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbors (KNN).

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Outlier Treatment](#outlier-treatment)
- [Data Scaling](#data-scaling)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Comparison](#model-comparison)
- [Forecasting Future Values](#forecasting-future-values)
- [How to Run the Code](#how-to-run-the-code)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

This project involves analyzing historical gold price data to predict the future `EUR/USD` exchange rate. The primary steps of the project include:

1. **Data Preprocessing**: Clean and scale the dataset for machine learning.
2. **Outlier Treatment**: Remove or handle outliers that could negatively impact model performance.
3. **Feature Engineering**: Create new features and scale the data to improve model performance.
4. **Model Training**: Train multiple regression models (Linear Regression, SVR, Decision Tree, Random Forest, and KNN).
5. **Model Evaluation**: Evaluate the models based on performance metrics such as Mean Absolute Error, Mean Squared Error, R-Squared, etc.
6. **Forecasting**: Predict `EUR/USD` values from 2019 to 2033.

## Data Preprocessing

1. **Missing Value Handling**: The dataset is cleaned to remove missing values using the `autoclean` function from the `datacleaner` library.
2. **Handling Duplicates**: Duplicated rows are removed to ensure data integrity.
3. **Feature Transformation**: Features are transformed to prepare the data for modeling.

## Outlier Treatment

Outliers are detected and treated using the **Interquartile Range (IQR)** method. Data points that are beyond the upper or lower whiskers are capped to ensure that the models aren't skewed by extreme values.

## Data Scaling

The data is scaled using the **QuantileTransformer** to ensure that all features have a uniform distribution. This helps improve the performance of models like Support Vector Regression (SVR) and K-Nearest Neighbors (KNN).

## Model Training and Evaluation

The following models are trained on the dataset:

- **Linear Regression (LR)**: A simple linear model to predict `EUR/USD`.
- **Support Vector Regression (SVR)**: A more advanced method for regression that can handle non-linear relationships.
- **Decision Tree Regressor (DTR)**: A tree-based model that splits the data into branches to make predictions.
- **Random Forest Regressor (RFR)**: An ensemble method that uses multiple decision trees to make predictions.
- **K-Nearest Neighbors (KNN)**: A non-parametric method that makes predictions based on the nearest neighbors of a data point.

Each model is evaluated using performance metrics such as Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-Squared.

## Model Comparison

The models are compared based on the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-Squared (R²)**
- **Mean Squared Error (MSE)**

This helps in understanding which model is the most accurate and suitable for predicting the gold price.

## Forecasting Future Values

Using the trained models, the `EUR/USD` exchange rate is predicted for the years 2019 to 2033. The predictions for these future years are visualized, comparing the predictions of multiple models (Linear Regression, SVR, Decision Tree, Random Forest, and KNN).
