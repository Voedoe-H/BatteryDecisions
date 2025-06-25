# Battery RUL Prediction with Decision Trees

This repository explores data-driven approaches for predicting **Remaining Useful Life (RUL)** of batteries using the [Battery Life Cycle dataset](https://www.kaggle.com/datasets/ignaciovinuales/battery-remaining-useful-life-rul/data) from Kaggle. It focuses on interpretable models such as **Decision Trees** and includes preprocessing, feature selection, training, and evaluation pipelines.

## Decision Tree Regression

### Predictions vs Actual
![Prediction Results](Figures/SimpleDecisionTreeRegressionPlot.png)

| Mean Squared Error | Mean Absolute Error | R2        |
|--------------------|---------------------|-----------|
| 8360.46            | 62.70               | 0.9195    |

## Random Forrest Regression

### Prediction vs Actual
![Prediction Results](Figures/RandomForrestRegressionPlot.png)

| Mean Squared Error | Mean Absolute Error | R2        |
|--------------------|---------------------|-----------|
| 5121.01            | 49.63               | 0.9507    |

## Gradient Boosting Regression

### Prediction vs Actual
![Prediction Results](Figures/GradientBoostingPlot.png)

| Mean Squared Error | Mean Absolute Error | R2        |
|--------------------|---------------------|-----------|
| 4644.84            | 50.40               | 0.9552    |