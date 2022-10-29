"""
IMPORTANT LIBRARIES

Pandas - loads data in 2D array format & performs analysis
Numpy - Used for computation, are faster than normal python arrays
Matplotlib/Seaborn - visualization library
Sklearn - Model development & evaluation, had pre-implemented functions
XGBoost - contains the eXtreme Gradient Boosting ML algo, that helps in achieving high accuracy in prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')


# Import Data
df = pd.read_csv('./data/TSLA.csv')
print(df.head())  # Note: Some data is missing, because on weekends & holidays, stock market is closed
print('*' * 100)
print(df.tail())


print(f'****************** Information on Data Rows & Columns: *************')
rows_and_columns = df.shape
print(rows_and_columns)


print(f'****************** More Information on data: *************')
more_data_info = df.describe()
print(more_data_info)
print(df.info())


# EXPLANATORY DATA ANALYSIS(EDA)
"""
EDA - An approach of analyzing data using visual techniques. Used to discover trends & patterns, with the help of
statistical summaries and graphical representation.
"""
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


# Lets do some data cleanup
# "Close" and "Adj Close" columns contain the same data, to test data
print(df[df['Close'] == df['Adj Close']].shape)

# remove the redundant data
df = df.drop(['Adj Close'], axis=1)


print(f'****************** Null values in the data frame *************')
df.isnull().sum()

# Distribution plot for continuous variable
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
    plt.show()


if __name__ == "__main__":
    pass
