import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import zipfile
import os
zip_file_path = '/content/house-prices-advanced-regression-techniques.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/extracted_data')
csv_file_path = os.path.join('/content/extracted_data', 'train.csv')
train_data = pd.read_csv(csv_file_path)
features = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
target = train_data['SalePrice']
features['TotalBathrooms'] = features['FullBath'] + 0.5 * features['HalfBath']
features = features.drop(columns=['FullBath', 'HalfBath'])
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")