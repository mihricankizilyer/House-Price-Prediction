
'''# Introduction
Content:

1. [Import Libraries & Modules](#1)
1. [Load Data](#2)
1. [Variable Description](#3)
    * [Categorical Variable Analysis](#4)
    * [Numerical Variable Analysis](#5)
1. [Target Analysis](#6)
1. [Correlation Analysis](#7)
1. [Feature Engineering](#8)
1. [Data Preprocessing](#9)
    * [Rare Encoding](#10)
    * [Label Encoding & One-Hot Encoding](#11)
    * [Missing Values](#12)
    * [Outliers](#13)
1. [Modeling](#14)
    * [Base Models](#15)
1. [Hyperparameter Optimization](#16)
1. [Feature Selection](#17)
1. [Hyperparameter Optimization with Selected Features](#18)
1. [Submission](#19)'''


# # 1. Import Libraries & Modules


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


######## REQUIRED INSTALL ########

# !pip install lightgbm
# !pip install catboost
# !pip install xgboost
# conda install -c conda-forge lightgbm
import sklearn
import seaborn as sns
import matplotlib.mlab as mlab

############ LIBRARIES ############

# BASE
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# DATA PREPROCESSING
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

# MODELING
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# MODEL TUNING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# # 2. Load Data

train = pd.read_csv("../input/house-price-dataset/train.csv")
test = pd.read_csv("../input/house-price-dataset/test.csv")

# train and test sets combined
df = train.append(test).reset_index(drop=True)


from shutil import copyfile

copyfile(src= "../input/helpers/data_prep.py", dst="../working/data_prep.py")
copyfile(src= "../input/helpers/eda.py", dst="../working/eda.py")

from data_prep import*
from eda import*

df.head()



# # 3. Variable Description

# #### *grab_col_name*:
# Returns the quantities of categorical, numeric, and categorical but cardinal variables in the data set.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_but_car

# The cardinal variable is not discarded because it has too many variables and how it carries the information is unknown.
df['Neighborhood'].value_counts()

# ## 3.1 Categorical Variable Analysis

for col in cat_cols:
    cat_summary(df, col)


for col in cat_but_car:
    cat_summary(df, col)

# ## 3.2 Numerical Variable Analysis
df[num_cols].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99]).T


# # 4. Target Analysis

df["SalePrice"].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T

# # 5. Correlation Analysis

# #### **find_correlation:**
# - will calculate negative or positive correlations greater than 60%
# - correlations of independent variables

def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

low_corrs, high_corrs = find_correlation(df, num_cols)

df.corr()["SalePrice"].sort_values(ascending=False).head(20)


# correlation between all variables

def target_correlation_matrix(dataframe, corr_th=0.5, target="SalePrice"):
    """
    Returns the variables that have a correlation above the threshold value given with the dependent variable..
    :param dataframe:
    :param corr_th: threshold value
    :param target:  dependent variable name
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("High threshold value, lower your corr_th value!")


target_correlation_matrix(df, corr_th=0.5, target="SalePrice")

#  ## 6.Feature Engineering
# MSZoning
# Identifies the general zoning classification of the sale.
# RH + RM = RM
df["MSZoning"].value_counts()
df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = "RM"


# df["LotArea"].mean() # 10168.11408016444
New_LotArea =  pd.Series(["studio_apartment","Small", "Middle", "Large","Dublex","Luxury Apartment"], dtype = "category")
df["New_LotArea"] = New_LotArea
df.loc[(df["LotArea"] > 35) & (df["LotArea"] <= 75), "New_LotArea"] = New_LotArea[0]
df.loc[(df["LotArea"] > 75) & (df["LotArea"] <= 200), "New_LotArea"] = New_LotArea[1]
df.loc[(df["LotArea"] > 200) & (df["LotArea"] <= 1000), "New_LotArea"] = New_LotArea[2]
df.loc[(df["LotArea"] > 1000) & (df["LotArea"] <= 3000), "New_LotArea"] = New_LotArea[3]
df.loc[(df["LotArea"] > 3000) & (df["LotArea"] <= 6000), "New_LotArea"] = New_LotArea[4]
df.loc[df["LotArea"] > 6000 ,"New_LotArea"] = New_LotArea[5]

# LotShape
# distance from the street connecting to the property
# aggregates into a single variable
df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"

# LandSlope: Slope of property
df.loc[(df["LandSlope"] == "Gtl"), "LandSlope"] = "Mod"

# Condition1: Proximity to various conditions
df.loc[(df["Condition1"] == "Feedr"),"Condition1"] = "Artery"
df.loc[(df["Condition1"] == "RRNn"),"Condition1"] = "RRAn"
df.loc[(df["Condition1"] == "RRNe"),"Condition1"] = "RRAn"
df.loc[(df["Condition1"] == "PosN"),"Condition1"] = "PosA"

# Condition2: Proximity to various conditions
df.loc[(df["Condition2"] == "RRNn"),"Condition1"] = "RRAn"
df.loc[(df["Condition2"] == "PosN"),"Condition1"] = "PosA"
df.loc[(df["Condition2"] == "RRNe"),"Condition1"] = "RRAe"

# HouseStyle: Style of dwelling
df.loc[(df["HouseStyle"] == "2.5Fin"), "HouseStyle"] = "2Story"
df.loc[(df["HouseStyle"] == "1.5Fin"), "HouseStyle"] = "1.5Unf"
df.loc[(df["HouseStyle"] == "2.5Fin"), "HouseStyle"] = "2.5Unf"

# Total number of bathrooms
df["new_total_bath"] = (df["BsmtFullBath"] + df["BsmtHalfBath"] + df["FullBath"] + df["HalfBath"])

# Assessment of the general condition
df["new_qual_cond"] = df["OverallQual"] + df["OverallCond"]

# Building age
df["new_built_remodadd"] = df["YearBuilt"] + df["YearRemodAdd"]

# Reviewed according to feature importance values
df["new_GrLivArea_LotArea"] = df["GrLivArea"] / df["LotArea"]
df["total_living_area"] = df["TotalBsmtSF"] + df["GrLivArea"]


# # 7. Data Preprocessing & Feature Engineering

# ## 7.1 Rare Encoding

# - It is an effort to consolidate the few classes.

# #### *rare_encoder* :
# - Examines the rare case of the entire cat_cols list, not just categorical
# - Fix if there is more than 1 rare
# - After the Rare class query is made according to 0.01, the sum of the trues is taken.
# - If it is greater than 1, it is included in the rare col list.

rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), "Rare", dataframe[col])
    return dataframe

df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]


cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)


# ## 7.2 Label Encoding & One-Hot Encoding

# The main purpose is to meet the demands of the algorithms and to eliminate the measurement problems that may occur or to produce a higher quality data.

cat_cols = cat_cols + cat_but_car
df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df[useless_cols_new].head()

for col in useless_cols_new:
    cat_summary(df, col)

# ## 7.3 Missing Values

missing_values_table(df)

test.shape

missing_values_table(train)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# ## 7.4 Outliers

for col in num_cols:
    print(col, check_outlier(df, col))

# # 8. Modeling

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)


y = np.log1p(train_df["SalePrice"])
X = train_df.drop(["Id","SalePrice"], axis=1)

# ## 8.1 Base Models

model = [("LightGBM", LGBMRegressor())]

for name, regressor in model:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# ## 9. Hyperparameter Optimization

# model object is entered
lgbm_model = LGBMRegressor(random_state=46)

# error before modeling (see 5-fold error with lgbn_model)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

# parameter set (parameter grid) is entered

lgbm_params = {"learning_rate": [0.01, 0.1, 0.05],
               "n_estimators": [1500, 3000, 6000],
               "colsample_bytree": [0.5, 0.7],
               "num_leaves": [31, 35],
               "max_depth": [3, 5]}

# traverse the parameter set and find the best combination
# decide which combinations of hyperparameters should be used
lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)
final_model


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

# # 10. Feature Selection

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# num=len(X) -> to be reviewed
plot_importance(final_model, X, 30)


feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})

num_summary(feature_imp, "Value", True)

feature_imp[feature_imp["Value"] >0].shape

feature_imp[feature_imp["Value"] <1].shape

# unnecessary variables
zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values

selected_cols = [col for col in X.columns if col not in zero_imp_cols]

len(selected_cols)

# # 11. Hyperparameter Optimization with Selected Features

lgbm_model = LGBMRegressor(random_state = 46)

lgbm_params = {"learning_rate": [0.01],
               "n_estimators": [3000],
               "colsample_bytree": [0.3, 0.8],
               "max_depth": [4],
               "num_leaves": [13],
               "min_child_samples": [3]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)
final_model


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
rmse

# # 12. Submission

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"].astype("Int32")

y_pred_sub = final_model.predict(test_df[selected_cols])

y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.to_csv('submission53.csv', index=False)

