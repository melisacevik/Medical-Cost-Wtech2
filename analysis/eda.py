import numpy as np
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import helpers.eda as eda

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)


df = pd.read_csv("dataset/insurance.csv")

# Overall Picture
eda.check_df(df)

# Disaggregation of variable types
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# Examination of categorical columns
for col in cat_cols:
    eda.cat_summary(df,col)

# Examination of numerical columns
df[num_cols].describe().T

# Correlation of numerical columns
eda.correlation_matrix(df,num_cols)

