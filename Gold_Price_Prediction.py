%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from fasteda import fast_eda
from datacleaner import autoclean

import scipy
import scipy.stats as stats

dataset = pd.read_csv("gold_price_data.csv")
dataset

dataset.info()


dataset.describe()

dataset.isnull().sum()

dataset.duplicated().sum()


dataset = autoclean(dataset)
dataset.head()


dataset.info()


fast_eda(dataset)


for column in dataset.columns : 
    plt.figure(figsize = (16,7))
    plt.subplot(121)
    sns.histplot(dataset[column])
    plt.title(column)
    
    plt.subplot(122)
    stats.probplot(dataset[column],dist = 'norm', plot = plt)
    plt.title(column)
    plt.show()




from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal')

for col in dataset.columns:
    dataset[col] = dataset[col] = qt.fit_transform(pd.DataFrame(dataset[col]))


for column in dataset.columns : 
    plt.figure(figsize = (16,7))
    plt.subplot(121)
    sns.histplot(dataset[column])
    plt.title(column)
    
    plt.subplot(122)
    stats.probplot(dataset[column],dist = 'norm', plot = plt)
    plt.title(column)
    plt.show()



# Create boxplots for each column
for column in dataset.columns:
    plt.figure(figsize=(20, 2))
    
    # Use Seaborn's boxplot to visualize the distribution
    sns.boxplot(data=dataset, x=column)
    
    # Set title for the boxplot
    plt.title(f'Boxplot of {column}')
    
    # Display the plot
    plt.show()


for col in dataset:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    whisker_width = 2.5
    lower_whisker = q1 - (whisker_width * iqr)
    upper_whisker = q3 + whisker_width * iqr
    dataset[col] = np.where(dataset[col] > upper_whisker, upper_whisker, np.where(dataset[col] < lower_whisker, lower_whisker, dataset[col]))

for column in dataset:
        plt.figure(figsize=(18,1.5))
        sns.boxplot(data = dataset, x = column)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tabulate import tabulate


# Extract features and target
X = dataset.drop("EUR/USD", axis = 1)
Y = dataset["EUR/USD"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



regLR = LinearRegression()
regLR.fit(X_train,Y_train)
yPredRegLR=regLR.predict(X_test)
print('mean_squared_error of Linear Regression=',mean_squared_error(yPredRegLR, Y_test))


regSVR=SVR(kernel='linear')
regSVR.fit(X_train,Y_train)
yPredRegSVR=regSVR.predict(X_test)
mean_squared_error(yPredRegSVR, Y_test)
print('mean_squared_error of SVR=',mean_squared_error(yPredRegSVR, Y_test))



regDTR=DecisionTreeRegressor()
regDTR.fit(X_train,Y_train)
yPredRegDTR=regDTR.predict(X_test)
print('mean_squared_error of Decision TREE Regression=',mean_squared_error(yPredRegDTR, Y_test))


regRFR=RandomForestRegressor()
regRFR.fit(X_train,Y_train)
yPredRegRFR=regRFR.predict(X_test)
print('mean_squared_error of Random Forest Regressor=',mean_squared_error(yPredRegRFR, Y_test))



regKNN = KNeighborsRegressor()
regKNN.fit(X_train, Y_train)
yPredRegKNN = regKNN.predict(X_test)
print('mean_squared_error of KNN=',mean_squared_error(yPredRegKNN, Y_test))




data=[
     ["Linear Regression",round(mean_absolute_error(yPredRegLR, Y_test),2),round(np.sqrt(mean_squared_error(yPredRegLR, Y_test)),2),round(r2_score(yPredRegLR, Y_test),2),round(mean_squared_error(yPredRegLR, Y_test),2)],
     ["Support Vector Regression",round(mean_absolute_error(yPredRegSVR, Y_test),2),round(np.sqrt(mean_squared_error(yPredRegSVR, Y_test)),2),round(r2_score(yPredRegSVR, Y_test),2),round(mean_squared_error(yPredRegSVR, Y_test),2)],
     ["Decision Tree Regression",round(mean_absolute_error(yPredRegDTR, Y_test),2),round(np.sqrt(mean_squared_error(yPredRegDTR, Y_test)),2),round(r2_score(yPredRegDTR, Y_test),2),round(mean_squared_error(yPredRegDTR, Y_test),2)],
     ["Random Forest Regression",round(mean_absolute_error(yPredRegRFR, Y_test),2),round(np.sqrt(mean_squared_error(yPredRegRFR, Y_test)),2),round(r2_score(yPredRegRFR, Y_test),2),round(mean_squared_error(yPredRegRFR, Y_test),2)], 
     ["K-Nearest Neighbors (KNN)", round(mean_absolute_error(yPredRegKNN, Y_test), 2), round(np.sqrt(mean_squared_error(yPredRegKNN, Y_test)), 2), round(r2_score(yPredRegKNN, Y_test), 2), round(mean_squared_error(yPredRegKNN, Y_test), 2)]
    ]
columns=["Model Name","Mean Absolute Error","Root Mean Squared Error","R Squared Error","Mean Squared Error"]

print(tabulate(data, headers=columns, tablefmt="fancy_grid"))



plt.scatter(Y_test, yPredRegLR, alpha=0.7, label='Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')


plt.title('Predicted vs. Actual Values(Linear Regression)')
plt.legend()
plt.show()



plt.scatter(Y_test, yPredRegSVR, alpha=0.7, label='Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Predicted vs. Actual Values(SVR)')
plt.legend()
plt.show()



plt.scatter(Y_test, yPredRegDTR, alpha=0.7, label='Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Predicted vs. Actual Values(DTR)')
plt.legend()
plt.show()


plt.scatter(Y_test, yPredRegRFR, alpha=0.7, label='Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Predicted vs. Actual Values(RFR)')
plt.legend()
plt.show()


plt.scatter(Y_test, yPredRegKNN, alpha=0.7, label='Predicted')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Predicted vs. Actual Values(KNN)')
plt.legend()
plt.show()


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto']  # Add more values if using 'rbf' or 'poly' kernels
}

# Create the GridSearchCV object
grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='neg_mean_squared_error')

# Fit the model to the training data
grid_search_svr.fit(X_train, Y_train)

# Get the best parameters
best_params_svr = grid_search_svr.best_params_

# Print the best parameters
print("Best Parameters for SVR:", best_params_svr)

# Predict using the tuned SVR model
regSVR_tuned = SVR(**best_params_svr)
regSVR_tuned.fit(X_train, Y_train)
yPredRegSVR_tuned = regSVR_tuned.predict(X_test)



data=[ 
    ["Support Vector Regression",round(mean_absolute_error(yPredRegSVR, Y_test),2),round(np.sqrt(mean_squared_error(yPredRegSVR, Y_test)),2),round(r2_score(yPredRegSVR, Y_test),2),round(mean_squared_error(yPredRegSVR, Y_test),2)], 
    ["SVR Tunned", round(mean_absolute_error(yPredRegSVR_tuned, Y_test), 2), round(np.sqrt(mean_squared_error(yPredRegSVR_tuned, Y_test)), 2), round(r2_score(yPredRegSVR_tuned, Y_test), 2), round(mean_squared_error(yPredRegSVR_tuned, Y_test), 2)]
    ]
columns=["Model Name","Mean Absolute Error","Root Mean Squared Error","R Squared Error","Mean Squared Error"]

print(tabulate(data, headers=columns, tablefmt="fancy_grid"))







# Plotting the actual vs. tuned predicted values for SVR
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, yPredRegSVR, color='green', label='SVR Predicted', alpha=0.7, marker='o')

plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Predicted Values for SVR')
plt.legend()
plt.show()



# Plotting the actual vs. tuned predicted values for SVR
plt.figure(figsize=(12, 6))

plt.scatter(Y_test, yPredRegSVR_tuned, color='blue', label='SVR Tuned Predicted', alpha=0.7, marker='x')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='orange', label='Perfect Prediction')

plt.title('Actual vs. Tuned Predicted Values for SVR')
plt.legend()
plt.show()






import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Replace 'your_actual_dataset_until_2018' with the name of your dataset
# Assume you have a dataset 'your_actual_dataset_until_2018' with columns 'Date', 'Feature1', 'Feature2', ..., 'EUR/USD'

# Extract features and target for training
X_train = dataset.drop("EUR/USD", axis=1)
Y_train = dataset["EUR/USD"]

# Create a new dataset for the years 2019 and above
# You can adjust the range of dates as needed
start_date_2019 = datetime(2019, 1, 1)
end_date_2033 = datetime(2033, 12, 31)
date_range_2019_above = pd.date_range(start=start_date_2019, end=end_date_2033, freq='D')

# Create a DataFrame with the new dates
dataset_2019_above = pd.DataFrame({"Date": date_range_2019_above})

# Convert 'Date' column to datetime64[ns]
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset_2019_above['Date'] = pd.to_datetime(dataset_2019_above['Date'])

# Use the same features as in the training set
dataset_2019_above = pd.merge(dataset_2019_above, dataset.drop("EUR/USD", axis=1), how='left', on='Date')

# Fill missing values with the mean or any other appropriate strategy
dataset_2019_above.fillna(dataset.mean(), inplace=True)

# Extract features for prediction
X_2019_above = dataset_2019_above.copy()  # Copy the entire DataFrame

# Drop the 'EUR/USD' column if it's not a feature
X_2019_above = X_2019_above.drop("EUR/USD", axis=1, errors='ignore')

# Apply QuantileTransformer to the features of the new dataset for 2019 and above
for col in X_2019_above.columns:
    X_2019_above[col] = qt.fit_transform(pd.DataFrame(X_2019_above[col]))

# Make predictions using the trained models for 2019 and above
yPredLR_2019_above = regLR.predict(X_2019_above)
yPredSVR_tuned_2019_above = regSVR_tuned.predict(X_2019_above)
yPredDTR_2019_above = regDTR.predict(X_2019_above)
yPredRFR_2019_above = regRFR.predict(X_2019_above)
yPredKNN_2019_above = regKNN.predict(X_2019_above)

# Plot the actual values until 2018 and predicted values for 2019 and above
plt.figure(figsize=(12, 6))

# Plot actual values until 2018
#plt.scatter(dataset['Date'], Y_train, color='blue', label='Actual Values until 2018', alpha=0.7, marker='o')
#plt.scatter(dataset['Date'], Y_train, color='blue', label='Actual Values until 2018', alpha=0.7, marker='o')

# Plot predicted values for 2019 and above
plt.plot(dataset_2019_above['Date'], yPredLR_2019_above, label='Linear Regression', linestyle='--')
plt.plot(dataset_2019_above['Date'], yPredSVR_tuned_2019_above, label='SVR Tuned', linestyle='--')
plt.plot(dataset_2019_above['Date'], yPredDTR_2019_above, label='Decision Tree Regression', linestyle='--')
plt.plot(dataset_2019_above['Date'], yPredRFR_2019_above, label='Random Forest Regression', linestyle='--')
plt.plot(dataset_2019_above['Date'], yPredKNN_2019_above, label='KNN', linestyle='--')

plt.xlabel('Date')
plt.ylabel('EUR/USD')
plt.title('Predicted Values(2019-33)')
plt.legend()
plt.show()





















