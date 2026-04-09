# 🏠 House Price Prediction

## 📌 Overview

In this project, I built an end-to-end machine learning pipeline to predict house prices using the Ames Housing dataset.

Rather than focusing only on model performance, I tried to understand how data preprocessing and feature engineering affect the final result.

> While working on this project, I realized that improving the data often matters more than trying more complex models.

---

## 🚀 Key Results

- RMSE: ~0.122 (CatBoost)  
- Kaggle Public Score: **0.12212**  
- Ensemble model provided slightly more stable predictions  

---

## 📊 Dataset

- 1460 training samples  
- 1459 test samples  
- 79 original features  

The dataset includes structural, quality, and location-based attributes of houses.

---

## 🔍 Exploratory Data Analysis

Before modeling, I analyzed the data to understand its structure and key relationships.
> This part helped me better understand how raw data behaves before modeling.

### Target Distribution

![Target](outputs/target_distribution.png)

The target variable is right-skewed, which can negatively affect model performance.

> At first, I underestimated how skewed the target variable was, but this turned out to be one of the most important factors affecting model performance.

---

### Log Transformation

![Log Target](outputs/log_target_distribution.png)

Log transformation stabilizes variance and improves learning.

---

### Correlation Analysis

![Correlation](outputs/correlation_heatmap.png)

Features like OverallQual and GrLivArea show strong correlation with price.

> One thing that stood out was how strongly overall quality and living area influenced the price. This directly guided my feature engineering decisions.

---

### Neighborhood Effect

![Neighborhood](outputs/neighborhood_vs_price.png)

Location has a significant impact on house prices.

---

### Quality vs Price

![Quality](outputs/overallqual_vs_price.png)

Higher quality houses clearly have higher prices.

---

### Living Area vs Price

![Area](outputs/grlivarea_vs_price.png)

Living area has a strong linear relationship with price.

---

### Missing Values

![Missing](outputs/missing_values.png)

Missing values are not random and require domain-based handling.

---

## 🧠 Feature Engineering

This was the most impactful part of the project.

> This was the part where I spent most of my time, and it had the biggest impact on performance.

I created new features to better represent the data:

- Total usable area  
- Quality × area interaction  
- House age & renovation age  
- Ratio-based features  
- Binary indicators (garage, basement, fireplace)

👉 The most important feature:  
**Quality × Living Area**

> Combining quality and living area into a single feature significantly improved the model, which showed me how powerful simple feature interactions can be.

---

## ⚙️ Data Preprocessing

- Missing values handled using domain knowledge  
- Outliers capped instead of removed  
- Rare categories grouped  
- Categorical variables encoded  
- Skewed numerical features transformed  

---

## 🤖 Modeling

I evaluated multiple models using cross-validation:

- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

CatBoost consistently gave the best results.

> Initially, I expected all boosting models to perform similarly, but CatBoost consistently gave better results across folds.
> Comparing different models helped me see how small changes in approach can affect performance.
---

## 🧠 Why CatBoost?

CatBoost performed better because:

- It handles complex tabular relationships well  
- It is more robust to overfitting  
- It provides stable predictions across folds  

---

## 🚀 Ensemble Learning

I combined:

- CatBoost  
- LightGBM  
- XGBoost  

Using weighted averaging.

> I didn’t expect a large improvement from the ensemble, but it helped make predictions slightly more stable.

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

- Feature engineering turned out to be more important than model choice  
- Proper preprocessing improved model stability  
- Ensemble methods provided small but consistent improvements  

---

## 🏁 Conclusion

This project made it clear that improving the data has a bigger impact than just trying more complex models.

Most of the performance gain came from feature engineering and proper preprocessing rather than model complexity.

> This project helped me understand that in real-world machine learning, the biggest improvements often come from better data representation rather than more complex algorithms.

---

## 🔗 Links

GitHub: https://github.com/rabia-ASIK/house-price-prediction  
Medium: https://medium.com/@rrabia.asik/beyond-baseline-designing-a-production-level-house-price-prediction-system-with-feature-0ce4f247aa78  
Kaggle: https://www.kaggle.com/rabiaas  

---

## 👩‍💻 Author

Rabia Aşık
