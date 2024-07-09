import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')
print(df.shape)

# Defining features and target/response variables
X = df.drop(['Class_Category'], axis=1) # df without Class_Category column
y = df[['Class_Category']] # only Class_Category column

# Train/test split in scikit-learn - 70% train， 30% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# xamining the shape of data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import numpy as np
print(np.mean(y_train))
print(np.mean(y_test))