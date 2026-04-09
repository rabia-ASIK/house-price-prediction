# 🏠 House Price Prediction

## 📌 Overview

In this project, I built a machine learning model to predict house prices using the Ames Housing dataset.

Rather than focusing only on model performance, I tried to understand how data preprocessing and feature engineering affect the results. The project follows a complete workflow from data analysis to model evaluation.

---

## 🚀 Key Results

- RMSE: ~0.122 (CatBoost)  
- Kaggle Public Score: 0.12212  
- Ensemble model slightly improved overall stability  

---

## 📊 Dataset

- 1460 training samples  
- 1459 test samples  
- 79 original features  

The dataset includes information about house structure, quality, and location.

---

## 🔍 Exploratory Data Analysis

Before modeling, I explored the dataset to understand its structure.

Main observations:

- The target variable was right-skewed  
- Log transformation improved distribution  
- Overall quality and living area were strongly correlated with price  
- Location (neighborhood) had a significant effect  

---

## 🧠 Feature Engineering

This was the most impactful part of the project.

I created new features to better represent the data, including:

- Total usable area  
- Quality × area interaction  
- House age and renovation age  
- Ratio-based features  
- Binary indicators (garage, basement, fireplace, etc.)

The interaction between quality and living area turned out to be the most important feature.

---

## ⚙️ Data Preprocessing

- Missing values handled based on feature meaning  
- Outliers capped instead of removed  
- Rare categories grouped  
- Categorical variables encoded  
- Skewed numerical features transformed  

---

## 🤖 Modeling

I tested several models:

- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

CatBoost consistently gave the best validation results.

---

## 🚀 Ensemble

I also combined CatBoost, LightGBM, and XGBoost using weighted averaging.

This provided a small improvement and made predictions more stable.

---

## 📈 Model Performance

| Model              | RMSE |
|-------------------|------|
| Random Forest     | ~0.138 |
| Gradient Boosting | ~0.125 |
| XGBoost           | ~0.135 |
| LightGBM          | ~0.131 |
| CatBoost          | ~0.119 |

---

## 📁 Outputs

- `submission_house_price_advanced.csv`  
- `submission_ensemble.csv`  
- EDA visualizations  
- Feature importance analysis  

---

## 💡 Key Takeaways

- Feature engineering had the biggest impact  
- Proper preprocessing improved model stability  
- Ensemble methods gave small but consistent gains  

---

## 🛠️ Technologies

- Python  
- Pandas / NumPy  
- Scikit-learn  
- CatBoost  
- LightGBM  
- XGBoost  

---

## 👩‍💻 Author

Rabia Aşık
