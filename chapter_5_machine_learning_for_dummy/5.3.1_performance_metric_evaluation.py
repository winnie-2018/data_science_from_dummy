import numpy as np
import pandas as pd
df = pd.read_csv('datasets/clean_creditcard.csv')

from sklearn.linear_model import LogisticRegression

lr_Object  = LogisticRegression(C=1.0, class_weight=None, dual=False,
                                fit_intercept=True, intercept_scaling=1,
                                max_iter=500, multi_class='auto', n_jobs=None,
                                penalty='l2', random_state=None,
                                solver='liblinear', tol=0.0001,
                                verbose=0, warm_start=False)


X = df.drop(['Class_Category'], axis=1)
y = df[['Class_Category']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



lr_Object.fit(X_train, y_train.values.ravel())

y_pred = lr_Object.predict(X_test)


#Classification Accuracy via Python
is_correct= y_pred==y_test.values.ravel()
print(f'Accuracy via Python: {np.mean(is_correct)}')
#--------------------------------

#Classification Accuracy via Scikit-learn
print(f'Accuracy via Scikit-learn: {lr_Object.score(X_test,y_test)}')

from sklearn import metrics
print(f'sklearn metrics: {metrics.accuracy_score(y_test,y_pred)}')
#--------------------------------
print()
# Calculate True and False Positive and Negative Rates
P = sum(y_test.values.ravel())
print(f'Positive Samples: {P}')

TP = sum( (y_test.values.ravel()==1) & (y_pred==1) )
print(f'True Positive: {TP}')

TPR = TP/P
print(f'positive Rate: {TPR}')

FN = sum( (y_test.values.ravel()==1) & (y_pred==0) )
print(f'False Negative: {FN}')

FNR = FN/P
print(f'False Negative rate: {FNR}')

N= sum(y_test.values.ravel()==0)
print(f'Negative: {N}')

TN= sum((y_test.values.ravel()==0) & (y_pred==0))
print(f'true Negative: {TN}')

FP = sum((y_test.values.ravel()==0) & (y_pred==1))
print(f'False Positive: {FP}')

TNR = TN/N
FPR = FP/N
print('the true negative rate is {} and the false positive rate is {}'. format(TNR,FPR))
#-----------------------------------------------

# The confusion matrix
print()
from sklearn.metrics import confusion_matrix
print(f"Confusion Matrix : \n {confusion_matrix(y_test, y_pred)}")
#--------------------------------

#Precision, Recall and F1 Score
Precision = TP/ (TP+FP)
print(f'Precision: {Precision}')

Recall = TP/ (TP+FN)
print(f'Recall: {Recall}')

F1Score = 2*((Precision * Recall)/(Precision+Recall))
print(f'F1Score: {F1Score}')
# #-----------------------------

# Classification Report
from sklearn.metrics import classification_report
print(f"Classification Report : \n {classification_report(y_test, y_pred)}")
#--------------------