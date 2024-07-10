import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

#  Import the LogisticRegression class and create an object from that class
from sklearn.linear_model import LogisticRegression

lr_Object  = LogisticRegression(C=1.0,
                                class_weight=None, 
                                dual=False,
                                fit_intercept=True, 
                                intercept_scaling=1,
                                max_iter=500, 
                                multi_class='auto', 
                                n_jobs=None,
                                penalty='l2', 
                                random_state=None,
                                solver='liblinear', 
                                tol=0.0001,
                                verbose=0, 
                                warm_start=False)

# Select the features and the response variables and get their data 
X = df.drop(['Class_Category'], axis=1)
y = df[['Class_Category']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model via calling the function .fit on the train data
lr_Object.fit(X_train, y_train.values.ravel())

y_pred = lr_Object.predict(X_test)
print(y_pred)