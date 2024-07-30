#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import shuffle 
import plotly.express as px
import os 
print("Current working directory:", os.getcwd())
from scipy.stats import chi2


# In[44]:


# Load the pre-split data 
train_data = pd.read_csv(r"C:\Users\Sherine\Documents\prodigy infotech\house-prices-advanced-regression-techniques\train.csv")
test_dataset = pd.read_csv(r"C:\Users\Sherine\Documents\prodigy infotech\house-prices-advanced-regression-techniques\test.csv")


# In[3]:


train_data.head()


# In[4]:


print(train_data.columns)


# In[5]:


features=['1stFlrSF','2ndFlrSF','GrLivArea','TotalBsmtSF','GarageArea','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','BedroomAbvGr','SalePrice']
sub_dataset= train_data[features] 
sub_dataset.head(1460)


# In[6]:


sub_dataset.isnull().sum()


# In[7]:


sub_dataset.loc[sub_dataset['BedroomAbvGr'] ==0 , 'BedroomAbvGr'] = np.nan 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sub_dataset.head(1460)


# In[8]:


sub_dataset['BedroomAbvGr'].isnull().sum()


# In[9]:


sub_dataset = sub_dataset.dropna(subset=['BedroomAbvGr'])
len(sub_dataset)


# In[10]:


sub_dataset['1stFlrSF'].isnull().sum()


# In[11]:


sub_dataset['TotalSF'] = (sub_dataset['1stFlrSF'] + sub_dataset['2ndFlrSF'] + sub_dataset['GrLivArea'] +
                   sub_dataset['TotalBsmtSF'] + sub_dataset['GarageArea'])

# Display the new column with the total square footage
print(sub_dataset[['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'TotalSF']].head(1460))


# In[12]:


sub_dataset['TotalBedroom'] = (sub_dataset['BedroomAbvGr'])

# Display the new column with the total square footage
print(sub_dataset[['BedroomAbvGr', 'TotalBedroom']].head(1460))


# In[13]:


sub_dataset['TotalBathrooms'] = (sub_dataset['FullBath'] +
                        sub_dataset['HalfBath'] * 0.5 +
                        sub_dataset['BsmtFullBath'] +
                        sub_dataset['BsmtHalfBath'] * 0.5)

# Verify the new feature
print(sub_dataset[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBathrooms']])


# In[14]:


print(sub_dataset[['TotalSF', 'BedroomAbvGr', 'TotalBathrooms', 'SalePrice']])


# In[15]:


# Correlation matrix
corr_matrix = sub_dataset.corr()
plt.figure(figsize=(15, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[16]:


sns.pairplot(sub_dataset[['TotalSF', 'BedroomAbvGr', 'TotalBathrooms', 'SalePrice']])
plt.show()


# In[17]:


fig = px.box(sub_dataset,x='TotalSF')
fig.show()


# In[25]:


# Assuming filtered_data is your DataFrame
new_sub_ds1 = pd.DataFrame(sub_dataset)

# Print column names to verify
print(new_sub_ds1.columns)

# Adjust the column names in value_vars based on the actual column names in new_sub_ds1
# For example, if the actual column names are 'TotalSF', 'Total_Bedrooms', 'Total_Bathrooms':
melted_df = pd.melt(new_sub_ds1, value_vars=[ 'TotalSF','TotalBedroom', 'TotalBathrooms'])

# Create a box plot
fig = px.box(melted_df, x='variable', y='value', title='Box Plot for Multiple Features')

# Show the plot
fig.show()


# In[49]:


def calculate_bounds(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# In[26]:


# Assuming filtered_data is your DataFrame
new_sub_ds1 = pd.DataFrame(sub_dataset)

# Print column names to verify
print(new_sub_ds1.columns)

# Adjust the column names in value_vars based on the actual column names in new_sub_ds1
# For example, if the actual column names are 'TotalSF', 'Total_Bedrooms', 'Total_Bathrooms':
melted_df = pd.melt(new_sub_ds1, value_vars=[ 'TotalBedroom', 'TotalBathrooms'])

# Create a box plot
fig = px.box(melted_df, x='variable', y='value', title='Box Plot for Multiple Features')

# Show the plot
fig.show()


# In[50]:


# Initialize a mask for non-outliers
non_outliers_mask = pd.Series([True] * len(sub_dataset), index=sub_dataset.index)

# Check each feature
features = ['TotalSF', 'TotalBedroom', 'TotalBathrooms']
for feature in features:
    lower, upper = calculate_bounds(sub_dataset, feature)
    feature_outliers = (sub_dataset[feature] < lower) | (sub_dataset[feature] > upper)
    non_outliers_mask &= ~feature_outliers

# Ensure the mask has the same index as the DataFrame
non_outliers_mask = non_outliers_mask.reindex(sub_dataset.index)

# Filter out outliers
filtered_data = sub_dataset.loc[non_outliers_mask]

# Display results
print("Filtered data (non-outliers):")
filtered_data


# In[60]:


# Step 1: Split the provided training data into training and validation sets
features = ['TotalSF', 'TotalBedroom', 'TotalBathrooms']
target = 'SalePrice'  # Assuming 'SalePrice' is your target variable

X = filtered_data[features]
y = filtered_data[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 


# In[61]:


model1 = LinearRegression()
model1.fit(X_train, y_train)


# In[62]:


# Predict on the test set
y_pred = model1.predict(X_val)

# Calculate performance metrics
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Percentage Error: {mape}%') 


# In[63]:


# Assuming test_dataset is your DataFrame with the raw features
test_dataset['TotalBathrooms'] = (
    test_dataset['FullBath'] +
    test_dataset['HalfBath'] * 0.4938 +
    test_dataset['BsmtFullBath'] +
    test_dataset['BsmtHalfBath'] * 0.4938
)

test_dataset['TotalSF'] = (
    test_dataset['1stFlrSF'] +
    test_dataset['2ndFlrSF'] +
    test_dataset['GrLivArea'] +
    test_dataset['TotalBsmtSF'] +
    test_dataset['GarageArea']
)

test_dataset['TotalBedroom'] = (test_dataset['BedroomAbvGr'])

# Define the features used for prediction
features = ['TotalSF', 'TotalBedroom', 'TotalBathrooms']
# Check for missing values in the test dataset features and handle if necessary
test_dataset = test_dataset.dropna(subset=features)

# Extract the features from the test dataset
X_test_final = test_dataset[features]

# Use the trained model to predict the sale prices
y_test_pred = model1.predict(X_test_final)

# Add predictions to the test dataset
test_dataset['Predicted_SalePrice'] = y_test_pred

# Display the test dataset with the predictions
test_dataset[['TotalSF', 'TotalBedroom', 'TotalBathrooms',  'Predicted_SalePrice']]


# In[68]:


# Add predictions to the test dataset
test_dataset['Predicted_SalePrice'] = y_test_pred

# Display the test dataset with the predictions
print(test_dataset[['TotalSF', 'TotalBedroom', 'TotalBathrooms', 'Predicted_SalePrice']].head())
model1_predictions = test_dataset[['TotalSF', 'TotalBedroom', 'TotalBathrooms']].copy()
model1_predictions['Predicted_SalePrice'] = y_test_pred
model1_predictions.to_excel('model1_predictions.xlsx' , index = False)


# In[ ]:




