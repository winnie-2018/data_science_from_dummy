import numpy as np
import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

## Building decision tree
from sklearn.tree import DecisionTreeClassifier

dt_object = DecisionTreeClassifier(max_depth=3)


X = df.drop(['Class_Category'], axis=1)
y = df[['Class_Category']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


dt_object.fit(X_train, y_train.values.ravel())

y_pred = dt_object.predict(X_test)

#Classification Accuracy via Python
is_correct= y_pred==y_test.values.ravel()
print(np.mean(is_correct))
#--------------------------------

#Classification Accuracy via Scikit-learn
print(dt_object.score(X_test,y_test))

from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
#--------------------------------

# Calculate True and False Positive and Negative Rates
P = sum(y_test.values.ravel())
print(P)

TP = sum( (y_test.values.ravel()==1) & (y_pred==1) )
print(TP)

TPR = TP/P
print(TPR)

FN = sum( (y_test.values.ravel()==1) & (y_pred==0) )
print(FN)

FNR = FN/P
print(FNR)

N= sum(y_test.values.ravel()==0)
print(N)

TN= sum((y_test.values.ravel()==0) & (y_pred==0))
print(TN)

FP = sum((y_test.values.ravel()==0) & (y_pred==1))
print(FP)

TNR = TN/N
FPR = FP/N
print('the true negative rate is {} and the false positive rate is {}'. format(TNR,FPR))
#-----------------------------------------------

# The confusion matrix
from sklearn.metrics import confusion_matrix
print(f"Confusion Matrix : \n {confusion_matrix(y_test, y_pred)}")
#--------------------------------

#Precision, Recall and F1 Score
Precision = TP/ (TP+FP)
print(Precision)

Recall = TP/ (TP+FN)
print(Recall)

F1Score = 2*((Precision * Recall)/(Precision+Recall))
print(F1Score)
#-----------------------------

# Classification Report
from sklearn.metrics import classification_report
print(f"Classification Report : \n {classification_report(y_test, y_pred)}")
#--------------------