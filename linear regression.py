from traceback import print_tb
import pandas as pd

df = pd.read_csv("/Users/TEMP/Documents/Personal/pythonProject/dataset.csv")
print(df.head())

import matplotlib.pyplot as plt
plt.scatter(df['Mileage'], df['Sell Price'])


plt.scatter(df['Age'], df['Sell Price'])


x = df[['Mileage', 'Age']]
y = df['Sell Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)
lin = LinearRegression()
lin.fit(x_train, y_train)

lin.predict(x_test)
print(lin.predict(x_test))
lin.score(x_test, y_test)
print(lin.score(x_test, y_test))



#x_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)
#x_test
#print(x_test)































