####################################################################################################
# HOUSE PRICE PREDICTION PROJECT
####################################################################################################

"""
PROJECT TITLE:
House Price Prediction with Advanced Feature Engineering and Ensemble Modeling

PROJECT DESCRIPTION:
This project aims to predict residential house prices in Ames, Iowa by using structural,
spatial, quality-related, and neighborhood-level property features.

BUSINESS PROBLEM:
Given a set of house characteristics, build a machine learning model that minimizes prediction
error and estimates house sale prices as accurately as possible.

DATASET:
- train.csv -> includes the target variable: SalePrice
- test.csv  -> SalePrice is missing and must be predicted

PROJECT GOALS:
1. Perform a complete exploratory data analysis (EDA)
2. Handle missing values and outliers carefully
3. Build powerful engineered features
4. Encode categorical variables properly
5. Compare multiple machine learning models
6. Improve performance using log transformation and hyperparameter tuning
7. Generate a Kaggle-ready submission file

WHY THIS PROJECT MATTERS:
This project demonstrates an end-to-end regression workflow and highlights the importance of:
- domain-aware preprocessing
- advanced feature engineering
- model selection
- optimization
- interpretable machine learning
"""

####################################################################################################
# 1. IMPORTS
####################################################################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

####################################################################################################
# 2. HELPER FUNCTIONS
####################################################################################################

def check_df(dataframe, head=5):
    """
    Displays the overall structure of the dataset.
    """
    print("##################### SHAPE #####################")
    print(dataframe.shape)
    print("##################### TYPES #####################")
    print(dataframe.dtypes)
    print("##################### HEAD #####################")
    print(dataframe.head(head))
    print("##################### TAIL #####################")
    print(dataframe.tail(head))
    print("##################### MISSING VALUES #####################")
    print(dataframe.isnull().sum().sort_values(ascending=False).head(20))
    print("##################### DESCRIBE #####################")
    print(dataframe.describe([0.01, 0.05, 0.50, 0.95, 0.99]).T)


def grab_col_names(dataframe, cat_th=20, car_th=30):
    """
    Returns categorical, numerical, and cardinal column names.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns
                if dataframe[col].dtype != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Categorical but cardinal: {len(cat_but_car)}")
    print(f"Numerical but categorical: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    """
    Summary for categorical variables.
    """
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("#############################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    """
    Summary for numerical variables.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("#############################################")

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Shows average target value by category.
    """
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "COUNT": dataframe[categorical_col].value_counts()
    }))
    print("#############################################")


def missing_values_table(dataframe, na_name=False):
    """
    Displays missing value counts and ratios.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)

    if na_name:
        return na_columns


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates lower and upper thresholds for outlier capping.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Checks whether a numerical variable contains outliers.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    """
    Caps outliers instead of removing them to avoid information loss.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


def rare_analyser(dataframe, target, cat_cols):
    """
    Examines rare categories and their target means.
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        }), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    """
    Groups infrequent categories under 'Rare'.
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns
                    if temp_df[col].dtype == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df


def label_encoder(dataframe, binary_col):
    """
    Label-encodes binary categorical variables.
    """
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """
    Applies one-hot encoding to multiclass categorical variables.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def plot_importance(model, features, num=20, save=False):
    """
    Plots feature importance for tree-based models.
    """
    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    }).sort_values("Value", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.head(num))
    plt.title("Feature Importance")
    plt.tight_layout()

    if save:
        plt.savefig("../outputs/feature_importance.png")

    plt.show()

####################################################################################################
# 3. TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
####################################################################################################

"""
GOAL:
Understand the dataset, identify variable types, inspect distributions,
check missing values and outliers, and explore relationships with the target variable.
"""

####################################################################################################
# Step 1: Read and combine train and test datasets
####################################################################################################

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train_rows = train.shape[0]
test_ids = test["Id"].copy()

# Add target column to test for consistent concatenation
test["SalePrice"] = np.nan

df = pd.concat([train, test], ignore_index=True)

# Keep Id separately for submission, remove it from modeling
df.drop("Id", axis=1, inplace=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Combined shape:", df.shape)

####################################################################################################
# Step 2: Identify numerical and categorical variables
####################################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

####################################################################################################
# Step 3: Apply required corrections (type issues etc.)
####################################################################################################

"""
MSSubClass is numeric in appearance but categorical in meaning.
It represents house class/type rather than a continuous quantity.
"""

df["MSSubClass"] = df["MSSubClass"].astype(str)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

####################################################################################################
# Step 4: Inspect distributions of numerical and categorical variables
####################################################################################################

# Optional detailed inspection
# for col in cat_cols:
#     if col != "SalePrice":
#         cat_summary(df, col)

# for col in num_cols:
#     if col != "SalePrice":
#         num_summary(df, col)

####################################################################################################
# Step 5: Analyze categorical variables with respect to the target
####################################################################################################

train_only = df[df["SalePrice"].notnull()].copy()

# Optional detailed target-category analysis
# for col in cat_cols:
#     if col != "SalePrice":
#         target_summary_with_cat(train_only, "SalePrice", col)

####################################################################################################
# Step 6: Check for outliers
####################################################################################################

print("\nOutlier check:")
for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

####################################################################################################
# Step 7: Check missing values
####################################################################################################

print("\nMissing values summary:")
na_cols = missing_values_table(df, na_name=True)
####################################################################################################
#EDA VISUALIZATIONS (All plots saved to outputs folder)
####################################################################################################

# 1. Distribution of target variable
plt.figure(figsize=(8, 5))
df["SalePrice"].dropna().hist(bins=100)
plt.title("SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../outputs/target_distribution.png")
plt.close()

# 2. Log-transformed target distribution
plt.figure(figsize=(8, 5))
np.log1p(df["SalePrice"].dropna()).hist(bins=60)
plt.title("Log(SalePrice) Distribution")
plt.xlabel("Log(SalePrice)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../outputs/log_target_distribution.png")
plt.close()

# 3. Correlation heatmap of numerical features
plt.figure(figsize=(14, 12))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap="RdBu", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../outputs/correlation_heatmap.png")
plt.close()

# 4. Top correlated features with target
corr_target = corr["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=corr_target.values, y=corr_target.index)
plt.title("Top Features Correlated with SalePrice")
plt.tight_layout()
plt.savefig("../outputs/top_correlated_features.png")
plt.close()

# 5. Neighborhood impact on house prices
plt.figure(figsize=(14, 6))
sns.boxplot(data=train_only, x="Neighborhood", y="SalePrice")
plt.xticks(rotation=90)
plt.title("Neighborhood vs SalePrice")
plt.tight_layout()
plt.savefig("../outputs/neighborhood_vs_price.png")
plt.close()

# 6. Effect of overall quality on price
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_only, x="OverallQual", y="SalePrice")
plt.title("Overall Quality vs SalePrice")
plt.tight_layout()
plt.savefig("../outputs/overallqual_vs_price.png")
plt.close()

# 7. Living area vs price relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(data=train_only, x="GrLivArea", y="SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.tight_layout()
plt.savefig("../outputs/grlivarea_vs_price.png")
plt.close()

# 8. Missing values visualization
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=missing.index, y=missing.values)
plt.xticks(rotation=90)
plt.title("Missing Values by Feature")
plt.tight_layout()
plt.savefig("../outputs/missing_values.png")
plt.close()

print("\nEDA visualizations saved to outputs folder.")

####################################################################################################
# 4. TASK 2: FEATURE ENGINEERING
####################################################################################################

"""
GOAL:
Improve the predictive power of the dataset by:
- handling missing values meaningfully
- capping outliers
- creating stronger features
- reducing category sparsity
- preparing variables for modeling
"""

####################################################################################################
# Step 1: Handle missing values and outliers
####################################################################################################

########################################
# 4.1 Outlier Treatment
########################################

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

print("\nOutlier check after capping:")
for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

########################################
# 4.2 Missing Value Treatment
########################################

"""
Missing values are handled based on variable meaning, not with a single blind rule.

Approach:
1. Missing values that indicate absence of a feature -> fill with explicit labels
2. Numerical absence-related variables -> fill with 0
3. Genuine missing values -> fill with mode / median / neighborhood-based median
"""

# Pool quality
if "PoolQC" in df.columns and "PoolArea" in df.columns:
    no_pool_rows = df["PoolArea"] == 0
    missing_pool_quality_rows = (df["PoolArea"] > 0) & (df["PoolQC"].isnull())

    df.loc[no_pool_rows, "PoolQC"] = "NoPool"
    df.loc[missing_pool_quality_rows, "PoolQC"] = "Unknown"

# Garage categorical variables
garage_cat_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
existing_garage_cat_cols = [col for col in garage_cat_cols if col in df.columns]

if len(existing_garage_cat_cols) > 0:
    garage_missing_rows = df[existing_garage_cat_cols].isnull().all(axis=1)
    for col in existing_garage_cat_cols:
        df.loc[garage_missing_rows, col] = "NoGarage"

# Garage numerical variables
garage_num_cols = ["GarageYrBlt", "GarageArea", "GarageCars"]
for col in garage_num_cols:
    if col in df.columns:
        if col == "GarageYrBlt":
            df[col] = df[col].fillna(df["YearBuilt"])
        else:
            df[col] = df[col].fillna(0)

# Basement categorical variables
bsmt_cat_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
existing_bsmt_cat_cols = [col for col in bsmt_cat_cols if col in df.columns]

if len(existing_bsmt_cat_cols) > 0:
    bsmt_missing_rows = df[existing_bsmt_cat_cols].isnull().all(axis=1)
    for col in existing_bsmt_cat_cols:
        df.loc[bsmt_missing_rows, col] = "NoBasement"

# Basement numerical variables
bsmt_num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
for col in bsmt_num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fireplace
if "FireplaceQu" in df.columns:
    df["FireplaceQu"] = df["FireplaceQu"].fillna("NoFireplace")

# Masonry veneer
if "MasVnrType" in df.columns:
    df["MasVnrType"] = df["MasVnrType"].fillna("None")

if "MasVnrArea" in df.columns:
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# Absence / unknown columns
for col in ["Alley", "Fence", "MiscFeature"]:
    if col in df.columns:
        df[col] = df[col].fillna("No")

# Neighborhood-based imputation for LotFrontage
if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

# Low-missing categorical columns -> mode
for col in ["Electrical", "KitchenQual", "Exterior1st", "Exterior2nd",
            "SaleType", "Functional", "Utilities", "MSZoning"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Remaining numeric missing values -> median
remaining_num_na = [col for col in df.columns
                    if df[col].dtype != "O" and df[col].isnull().sum() > 0 and col != "SalePrice"]

for col in remaining_num_na:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after treatment:")
missing_values_table(df)
# Final safeguard: fill any remaining categorical missing values with mode
remaining_cat_na = [col for col in df.columns if df[col].dtype == "O" and df[col].isnull().sum() > 0]
for col in remaining_cat_na:
    df[col] = df[col].fillna(df[col].mode()[0])

# Final safeguard: fill any remaining numerical missing values with median
remaining_num_na = [col for col in df.columns if df[col].dtype != "O" and df[col].isnull().sum() > 0 and col != "SalePrice"]
for col in remaining_num_na:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after final safeguard:")
missing_values_table(df)

####################################################################################################
# Step 2: Perform rare analysis and apply rare encoder
####################################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
rare_cols = [col for col in cat_cols if col != "SalePrice"]

# Optional inspection
# rare_analyser(df[df["SalePrice"].notnull()], "SalePrice", rare_cols)

df = rare_encoder(df, 0.01)

####################################################################################################
# Step 3: Create new features
####################################################################################################

"""
This is one of the most critical sections of the project.

To maximize predictive performance, feature engineering combines:
- the instructor's stronger area/ratio/quality-based ideas
- your additional practical features
- extra interpretable binary and interaction features

These engineered variables are designed to better capture:
- house size
- usable living space
- relative scale
- quality-density interaction
- renovation dynamics
- convenience-related signals
- structural presence/absence
"""

########################################
# A. AREA-BASED FEATURES
########################################

# Total above-ground floor area
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total finished basement area
df["NEW_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

# Total porch / deck / outdoor living area
df["NEW_PorchArea"] = (
    df["OpenPorchSF"] +
    df["EnclosedPorch"] +
    df["ScreenPorch"] +
    df["3SsnPorch"] +
    df["WoodDeckSF"]
)

# Total house area including basement
df["NEW_TotalHouseArea"] = df["NEW_TotalFlrSF"] + df["TotalBsmtSF"]

# Total usable square footage
df["NEW_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]

# Total indoor + outdoor combined area
df["NEW_TotalPropertyUseArea"] = df["NEW_TotalHouseArea"] + df["NEW_PorchArea"]

########################################
# B. MULTIPLICATIVE / INTERACTION FEATURES
########################################

# First floor area weighted by living area
df["NEW_1st_GrLiv_MUL"] = df["1stFlrSF"] * df["GrLivArea"]

# Garage area weighted by living area
df["NEW_Garage_GrLiv_MUL"] = df["GarageArea"] * df["GrLivArea"]

# Overall quality combined with condition
df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]

# Quality-area interaction
df["NEW_Qual_x_Area"] = df["OverallQual"] * df["GrLivArea"]

# Condition-area interaction
df["NEW_Cond_x_Area"] = df["OverallCond"] * df["GrLivArea"]

# Garage capacity weighted by garage area
df["NEW_GarageCapacityScore"] = df["GarageCars"] * df["GarageArea"]

########################################
# C. RATIO FEATURES
########################################

# Living area relative to lot area
df["NEW_LotRatio"] = df["GrLivArea"] / (df["LotArea"] + 1)

# Total house area relative to lot area
df["NEW_RatioArea"] = df["NEW_TotalHouseArea"] / (df["LotArea"] + 1)

# Garage area relative to lot area
df["NEW_GarageLotRatio"] = df["GarageArea"] / (df["LotArea"] + 1)

# Masonry veneer area relative to house size
df["NEW_MasVnrRatio"] = df["MasVnrArea"] / (df["NEW_TotalHouseArea"] + 1)

# Porch area relative to lot area
df["NEW_PorchLotRatio"] = df["NEW_PorchArea"] / (df["LotArea"] + 1)

# Basement area relative to total house area
df["NEW_BsmtRatio"] = df["TotalBsmtSF"] / (df["NEW_TotalHouseArea"] + 1)

# Garage area relative to total house area
df["NEW_GarageHouseRatio"] = df["GarageArea"] / (df["NEW_TotalHouseArea"] + 1)

########################################
# D. DIFFERENCE / UNUSED SPACE FEATURES
########################################

# Remaining open/unbuilt lot area
df["NEW_DifArea"] = (
    df["LotArea"] -
    df["1stFlrSF"] -
    df["GarageArea"] -
    df["NEW_PorchArea"]
)

# Difference between total house area and above-ground living area
df["NEW_HouseLivDiff"] = df["NEW_TotalHouseArea"] - df["GrLivArea"]

########################################
# E. AGE / TIME FEATURES
########################################

# Years between remodel and original build
df["NEW_Restoration"] = df["YearRemodAdd"] - df["YearBuilt"]

# House age at sale
df["NEW_HouseAge"] = df["YrSold"] - df["YearBuilt"]

# Years since remodel at sale
df["NEW_RestorationAge"] = df["YrSold"] - df["YearRemodAdd"]

# Gap between garage build year and house build year
df["NEW_GarageAge"] = df["GarageYrBlt"] - df["YearBuilt"]

# Absolute difference between garage year and remodel year
df["NEW_GarageRestorationAge"] = np.abs(df["GarageYrBlt"] - df["YearRemodAdd"])

# Years since garage construction at sale
df["NEW_GarageSold"] = df["YrSold"] - df["GarageYrBlt"]

# Is the property relatively new?
df["NEW_IsNewHouse"] = np.where(df["NEW_HouseAge"] <= 10, 1, 0)

########################################
# F. BATHROOM / ROOM EFFICIENCY FEATURES
########################################

# Weighted total bathroom count
df["NEW_TotalBath"] = (
    df["FullBath"] +
    0.5 * df["HalfBath"] +
    df["BsmtFullBath"] +
    0.5 * df["BsmtHalfBath"]
)

# Average room size
df["NEW_AvgRoomSqFt"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

# Bathrooms per room
df["NEW_BathPerRoom"] = df["NEW_TotalBath"] / (df["TotRmsAbvGrd"] + 1)

# Bedrooms proportion within total rooms
df["NEW_BedroomRatio"] = df["BedroomAbvGr"] / (df["TotRmsAbvGrd"] + 1)

########################################
# G. BINARY PRESENCE FEATURES
########################################

# Fireplace presence
df["NEW_HasFireplace"] = np.where(df["Fireplaces"] > 0, 1, 0)

# Garage presence
df["NEW_HasGarage"] = np.where(df["GarageArea"] > 0, 1, 0)

# Basement presence
df["NEW_HasBsmt"] = np.where(df["TotalBsmtSF"] > 0, 1, 0)

# Second floor presence
df["NEW_Has2ndFloor"] = np.where(df["2ndFlrSF"] > 0, 1, 0)

# Pool presence
df["NEW_HasPool"] = np.where(df["PoolArea"] > 0, 1, 0)

# Porch presence
df["NEW_HasPorch"] = np.where(df["NEW_PorchArea"] > 0, 1, 0)

# Masonry veneer presence
df["NEW_HasMasVnr"] = np.where(df["MasVnrArea"] > 0, 1, 0)

# Central air presence as binary consistency feature
if "CentralAir" in df.columns:
    df["NEW_HasCentralAir"] = np.where(df["CentralAir"] == "Y", 1, 0)

########################################
# H. REMODEL / QUALITY FLAGS
########################################

# Whether the house has been remodeled
df["NEW_IsRemodeled"] = np.where(df["YearBuilt"] != df["YearRemodAdd"], 1, 0)

# High quality house flag
df["NEW_HighQual"] = np.where(df["OverallQual"] >= 7, 1, 0)

# High condition house flag
df["NEW_HighCond"] = np.where(df["OverallCond"] >= 6, 1, 0)

########################################
# I. SEASONAL / SALE-TIME FEATURES
########################################

# Convert month sold into season-like bins
if "MoSold" in df.columns:
    df["NEW_SoldInSpring"] = np.where(df["MoSold"].isin([3, 4, 5]), 1, 0)
    df["NEW_SoldInSummer"] = np.where(df["MoSold"].isin([6, 7, 8]), 1, 0)

####################################################################################################
# Step 4: Perform encoding
####################################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
ohe_cols = [col for col in cat_cols if col != "SalePrice" and col not in binary_cols]

df = one_hot_encoder(df, ohe_cols, drop_first=True)

print("\nShape after encoding:", df.shape)
####################################################################################################
# ADDITIONAL SKEWNESS CORRECTION
####################################################################################################

train_df_temp = df[df["SalePrice"].notnull()].copy()

numeric_cols_for_skew = [col for col in df.columns if df[col].dtype != "O" and col != "SalePrice"]

skewness = train_df_temp[numeric_cols_for_skew].skew().sort_values(ascending=False)
skewed_cols = skewness[skewness > 0.75].index.tolist()

for col in skewed_cols:
    if (df[col] >= 0).all():
        df[col] = np.log1p(df[col])

print("\nSkewed feature sayısı:", len(skewed_cols))

####################################################################################################
# 5. TASK 3: MODELING
####################################################################################################

"""
GOAL:
Split train and test sets again, compare multiple models,
select the strongest one, optimize it, and generate final predictions.
"""

####################################################################################################
# Step 1: Split train and test data
####################################################################################################

train_df = df[df["SalePrice"].notnull()].copy()
test_df = df[df["SalePrice"].isnull()].copy()

X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]

# Log-transformed target for improved learning stability
y_log = np.log1p(train_df["SalePrice"])

X_test = test_df.drop("SalePrice", axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)

####################################################################################################
# Step 2: Train baseline models and evaluate performance
####################################################################################################

"""
Multiple models are compared using 5-fold cross-validation on log-transformed target.
Lower RMSE indicates better performance.
"""

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror"),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=42)
}

print("\nModel comparison results (5-Fold CV, RMSE on log scale):")
for name, model in models.items():
    rmse = np.sqrt(-cross_val_score(model, X, y_log, cv=5, scoring="neg_mean_squared_error")).mean()
    print(f"{name}: {rmse:.4f}")

####################################################################################################
# Bonus: Apply log transformation to the target
####################################################################################################

"""
SalePrice is right-skewed.
Log transformation improves:
- stability
- learning behavior
- sensitivity to extreme prices
"""

####################################################################################################
# Step 3: Hyperparameter optimization
####################################################################################################

"""
CatBoost is selected as the final candidate model due to its strong performance
and robustness on tabular mixed-type data.
"""

catboost_model = CatBoostRegressor(verbose=False, random_state=42)

catboost_params = {
    "iterations": [400, 600],
    "learning_rate": [0.03, 0.05],
    "depth": [4, 6],
    "l2_leaf_reg": [3, 5]
}

catboost_best_grid = GridSearchCV(
    estimator=catboost_model,
    param_grid=catboost_params,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring="neg_mean_squared_error"
)

catboost_best_grid.fit(X, y_log)

print("\nBest parameters:", catboost_best_grid.best_params_)
print("Best CV RMSE:", np.sqrt(-catboost_best_grid.best_score_))

####################################################################################################
# Final Model Training
####################################################################################################

final_model = CatBoostRegressor(
    **catboost_best_grid.best_params_,
    verbose=False,
    random_state=42
)

final_model.fit(X, y_log)
####################################################################################################
# ADVANCED ENSEMBLE MODELING
####################################################################################################

xgb_final = XGBRegressor(
    random_state=42,
    objective="reg:squarederror",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8
)

lgbm_final = LGBMRegressor(
    random_state=42,
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1
)

catboost_final = CatBoostRegressor(
    **catboost_best_grid.best_params_,
    verbose=False,
    random_state=42
)

print("\nEnsemble CV sonuçları:")

for name, model in {
    "CatBoost": catboost_final,
    "LightGBM": lgbm_final,
    "XGBoost": xgb_final
}.items():
    rmse = np.sqrt(-cross_val_score(model, X, y_log, cv=5, scoring="neg_mean_squared_error")).mean()
    print(f"{name}: {rmse:.4f}")

# Fit
catboost_final.fit(X, y_log)
lgbm_final.fit(X, y_log)
xgb_final.fit(X, y_log)

# Predict
cat_pred = catboost_final.predict(X_test)
lgb_pred = lgbm_final.predict(X_test)
xgb_pred = xgb_final.predict(X_test)

# Weighted average
ensemble_pred_log = (
    0.70 * cat_pred +
    0.15 * xgb_pred +
    0.15 * lgb_pred
)

ensemble_pred = np.expm1(ensemble_pred_log)

submission_ensemble = pd.DataFrame({
    "Id": test_ids.astype(int),
    "SalePrice": ensemble_pred
})

submission_ensemble.to_csv("../outputs/submission_ensemble.csv", index=False)

print("\nEnsemble submission oluşturuldu!")
print(submission_ensemble.head())

####################################################################################################
# Step 4: Examine feature importance
####################################################################################################

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 20 Feature Importances:")
print(feature_importance.head(20))

plot_importance(final_model, X, num=20, save=False)

####################################################################################################
# Bonus: Predict missing SalePrice values in test set and create Kaggle submission file
####################################################################################################

test_preds_log = final_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({
    "Id": test_ids.astype(int),
    "SalePrice": test_preds
})

submission.to_csv("../outputs/submission_house_price_advanced.csv", index=False)

print("\nSubmission file created successfully: submission_house_price_advanced.csv")
print(submission.head())

####################################################################################################
# PROJECT SUMMARY
####################################################################################################

"""
FINAL SUMMARY:
- Combined train and test sets for consistent preprocessing
- Performed EDA to understand feature distributions and target behavior
- Treated missing values based on semantic meaning
- Capped outliers to preserve information while reducing distortion
- Applied rare encoding to sparse categorical levels
- Built a rich feature engineering layer including area, ratio, age, interaction,
  efficiency, binary presence, and renovation-based features
- Encoded categorical variables using label encoding and one-hot encoding
- Compared multiple strong models
- Used log transformation on SalePrice
- Tuned CatBoost via GridSearchCV
- Examined feature importance
- Generated Kaggle-ready predictions

PROJECT STRENGTH:
This solution combines domain understanding, robust preprocessing,
advanced feature engineering, strong ensemble learning, and clear interpretability.
"""