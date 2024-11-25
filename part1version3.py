import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Step 2: Explore data (optional for debugging)
print(train.info())
print(test.info())

# Step 3: Correlation Analysis
# Select numeric columns only
numeric_data = train.select_dtypes(include=['number'])

# Compute correlation with SalePrice
correlation = numeric_data.corr()
corr_with_price = correlation['SalePrice'].sort_values(ascending=False)
print("Top correlated features:\n", corr_with_price.head(10))

# Plot heatmap for top features
top_features = corr_with_price.head(10).index
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data[top_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top 10 Correlated Features with SalePrice")
plt.show()


# Step 4: Handle Missing Values (for key features in train and test)
# Fill missing numerical values with the median
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

# Fill missing categorical values with the mode
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

# Drop columns with excessive missing data
columns_to_drop = ['Alley', 'PoolQC', 'MiscFeature', 'Fence']
train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 5: Feature Selection
selected_features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'OverallQual', 'MSZoning']
X = train[selected_features]
y = train['SalePrice']

# Step 6: Build Preprocessing Pipeline
# Separate numeric and categorical features
numeric_features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['MSZoning']

# Define transformations for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 7: Train-Test Split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Define and Train Model
# Option 1: Linear Regression
model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

model_lr.fit(X_train_split, y_train_split)

# Evaluate Linear Regression
y_val_pred_lr = model_lr.predict(X_val_split)
mae_lr = mean_absolute_error(y_val_split, y_val_pred_lr)
r2_lr = r2_score(y_val_split, y_val_pred_lr)
print(f"Linear Regression - MAE: {mae_lr}, R²: {r2_lr}")



# Step 9: Predict on Test Data
X_test = test[selected_features]
test['SalePrice'] = model_lr.predict(X_test)

#plot actual vs predicted prices

plt.scatter(y_val_split, y_val_pred_lr, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Validation Set)')
plt.show()

#Visualiziing the price distribution 
plt.hist(train['SalePrice'], bins=50, alpha=0.5, label='Actual Prices')
plt.hist(test['SalePrice'], bins=50, alpha=0.5, label='Predicted Prices')
plt.legend()
plt.title('Actual vs. Predicted Price Distributions')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#print statistics for training and predicted prices
print("Training SalePrice Statistics:")
print(train['SalePrice'].describe())
print("\nPredicted SalePrice Statistics:")
print(test['SalePrice'].describe())

#Checking the residuals to see if errors are randomly distributed:
residuals = y_val_split - y_val_pred_lr
plt.scatter(y_val_pred_lr, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Step 10: Save Predictions for Submission
submission = test[['Id', 'SalePrice']]
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
