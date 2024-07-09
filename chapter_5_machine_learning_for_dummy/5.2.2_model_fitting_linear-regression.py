# Generating synthetic data
from numpy.random import default_rng
rg = default_rng(12345) # set seeds
X = rg.uniform(low=0.0, high=20.0, size=(2000,)) # print 2000 elem
print(X[0:10])

#Data for linear regression (response variable)
slope = 0.2
intercept = -1.2
y = slope * X + intercept + rg.normal(loc=0.0, scale=1.0, size=(2000,)) # y=mx+c + error

#Linear regression modeling
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
print(lin_reg) # Output; LinearRegression()
 
lin_reg.fit(X.reshape(-1,1), y) # To fit the model, find CoE
print(lin_reg.intercept_)
print(lin_reg.coef_)

y_pred = lin_reg.predict(X.reshape(-1,1))

import matplotlib.pyplot as plt #import plotting package
import matplotlib as mpl #additional plotting functionality and parameters
mpl.rcParams['figure.dpi'] = 250
plt.scatter(X,y, s=1)
plt.plot(X,y_pred,'g')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('fig')
plt.show()