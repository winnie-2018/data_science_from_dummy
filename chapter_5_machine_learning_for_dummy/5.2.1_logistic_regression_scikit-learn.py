from sklearn.linear_model import LogisticRegression
lr_Object = LogisticRegression()
print(lr_Object)

new_lr_Object  = LogisticRegression(
        penalty='l2', dual=False,
        tol=0.0001, C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver='lbfgs',
        max_iter=100,
        multi_class='auto',
        verbose=0, 
        warm_start=False,
        n_jobs=None, 
        l1_ratio=None
    )

new_lr_Object.C = 0.2
new_lr_Object.solver = 'liblinear'
new_lr_Object.max_iter= 500
print(new_lr_Object)

import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

X = df[0:700].values
y = df['Class_Category'][0:700].values
new_lr_Object.fit(X, y)

new_X = df[700:715].values
print(new_X)

y_pred = new_lr_Object.predict(new_X)
print(f'Predicted values: {y_pred}')

y_test= df['Class_Category'][700:715].values
print(f'Actual values: {y_test}')
