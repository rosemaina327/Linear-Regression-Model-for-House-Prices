##loading the dat using pandas
import pandas as pd


#Load datasets
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

#view basic information
print(train.info()) #prints out all the entries ,columns ,columns name ,dta type and what columns have missing values
print(train.head()) #prints the first five rows of the trains.csv dataset.Helps confirm the structure and content of the data
print(test.info()) #should miss as expected

#Count the missing values
print(train.isnull().sum())
print(test.isnull().sum())

#inspect feature correlations to find a correlation matrix to find features that are strongly correlated with saleprices

import seaborn as sns
import matplotlib.pyplot as plt

#select numeric columns only
numeric_data = train.select_dtypes(include=['number'])

#compute correlation matrix
correlation = numeric_data.corr()

#visualize the correlation matrix
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.show()  #look for features with a hgh correlation to saleprices (e.g., GrLivArea, OverallQual, TotalBsmtSF)


#fiiling the missing values with the mean median or fixed value for numerical values
train['LotFrontage'].fillna(train['LotFrontage'].median(),inplace = True)
test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)

#filing categorical features with the mode(most frequent value)
train['MSZoning'].fillna(train['MSZoning'].mode()[0], inplace =True)
test['MSZoning'].fillna(train['MSZoning'].mode()[0], inplace =True)

#check whether MiscFeature is present in test.csv before dropping
test.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1, inplace=True, errors='ignore')


#drrop features with excessive missing value like ally and PoolQC
train.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1, inplace=True)
test.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1, inplace=True)

#training selected features for the Model
selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageArea']
X_train =  train[selected_features]
y_train =  train['SalePrice']
X_test = test[selected_features]

#Normalize or standardize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#split the train dataset for training and validation
from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size = 0.2,random_state = 42)

#training a linear regression model:
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_split, y_train_split)

#evaluate the model
from sklearn.metrics import mean_absolute_error

y_val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
print(f'Mean Absolute Error: {mae}') #mae-mean absolute error

#make predictions for the test dataset
y_test_pred = model.predict(X_test)

import numpy as np
submission = pd.DataFrame({'Id':test['Id'], 'SalePrice' : np.round(y_test_pred, 2)})
submission.to_csv('submission.csv', index = False)



