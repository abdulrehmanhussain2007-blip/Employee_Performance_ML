import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

matplotlib.use('TkAgg')

df=pd.read_csv("smart_workforce_productivity.csv")
print(df.head())
print(df.columns)
print(df.describe)
x = df.drop("productivity_score", axis=1)
print(x)
y = df["productivity_score"]
print(y)

sns.pairplot(df)
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=5,
)

from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln.fit(x_train,y_train)
print(ln.intercept_)
print(ln.coef_)

predictions=ln.predict(x_test)
print(predictions)

from sklearn.metrics import explained_variance_score,mean_absolute_error,confusion_matrix
print("explained_variance_score",explained_variance_score(y_test,predictions))
print("mean_absolute_error",mean_absolute_error(y_test,predictions))