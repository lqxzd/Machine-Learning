# coding:utf-8

# 多元线性回归
from pyexpat import model
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

x_train = np.array([[2,4],[5,8],[5,9],[7,10],[9,12]])
y_train = np.array([20,50,30,70,60])

# 训练
model.fit(x_train,y_train)

print(model.coef_)                  # 输出系数w
print(model.intercept_)             # 输出截距b
print(model.score(x_train,y_train)) # 输出训练结果