import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Select relevant features
selected_features = ['GrLivArea', 'OverallQual', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'BedroomAbvGr']
X_train = train[selected_features]
y_train = train['SalePrice']
X_test = test[selected_features]

# Train-test split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_split = scaler.fit_transform(X_train_split)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Evaluate the model
y_val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict for the test set
y_test_pred = model.predict(X_test)

# Save predictions to a CSV
output = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_test_pred})
output.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")
