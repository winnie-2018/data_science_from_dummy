import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

## Building decision tree
from sklearn.tree import DecisionTreeClassifier
dt_object = DecisionTreeClassifier(max_depth=3)
print(dt_object)

X = df.drop(['Class_Category'], axis=1)
y = df[['Class_Category']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


dt_object.fit(X_train, y_train.values.ravel())

y_pred = dt_object.predict(X_test)
y_pred







