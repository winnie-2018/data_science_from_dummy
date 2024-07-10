import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

from sklearn.ensemble import RandomForestClassifier
rf_obj = RandomForestClassifier(n_estimators=200)

X = df.drop(['Class_Category'], axis=1)
y = df[['Class_Category']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


rf_obj.fit(X_train, y_train.values.ravel())

y_pred = rf_obj.predict(X_test)
print(y_pred)