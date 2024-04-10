import numpy as np
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler

import helpers.data_preprocessing as data_preprocess
import helpers.eda as eda
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv("dataset/insurance.csv")

# Setting thresholds for outliers : outlier_thresholds()

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# Check for outliers in columns

for col in num_cols:
    print(data_preprocess.check_outlier(df,col))

# Output : no outlier observations

# missing values

df.isnull().sum()

# Result : no missing values

df.columns = [col.upper() for col in df.columns ]
df.columns

# New feature

#AGE
df.loc[(df["AGE"] <35), "NEW_AGE_CAT"] = "young"
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

#BMI
# BMI
df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                             labels=["underweight", "healty", "overweight", "obese"])
df.tail()

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df, cat_th=10, car_th=20)

for col in cat_cols:
    eda.cat_summary(df, col)

df.columns

for col in cat_cols:
    eda.target_summary_with_cat(df, "CHARGES", col)

# GÖRSELLEŞTİR BURAYI

df.info()

# label encoder
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    data_preprocess.label_encoder(df, col)

data_preprocess.label_encoder(df,"SEX")

df.head()

#one - hot encoder

one_hot = ["REGION","NEW_AGE_CAT","NEW_BMI_RANGE"]
df = pd.get_dummies(df, columns=one_hot, drop_first=True,dtype=int)

df.head()
# Scaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop("CHARGES", axis=1))
df_scaled = pd.DataFrame(df_scaled, columns=df.drop("CHARGES", axis=1).columns)


# Modelling

X = df_scaled  # Bağımsız değişkenler
y = df['CHARGES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_test.shape
X_train.shape
y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)
reg_model.intercept_ #sabit
reg_model.coef_ #bağımsız değişkenkerin ağırlıkları

y_pred = reg_model.predict(X)

print(reg_model.score(X_test,y_test))

# quad

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))


# random forest

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'squared_error',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(X_train,y_train)
forest_train_pred = forest.predict(X_train)
forest_test_pred = forest.predict(X_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

# sonuç :
# MSE train data: 3537636.561, MSE test data: 22724220.835
# R2 train data: 0.976, R2 test data: 0.848
