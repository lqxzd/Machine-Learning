# coding:utf-8

# 一元线性回归
import matplotlib.pyplot as plt
import numpy as np

x_train = [4,8,5,10,12]
y_train = [20,50,30,70,60]

# 画图函数
def draw(x_train,y_train):
    plt.scatter(x_train,y_train)

# 定义函数求得斜率w和截距b
# 使用最小二乘法对斜率和截距求导并使得导数值等于0求解出斜率和截距
def fit(x_train,y_train):
    size = len(x_train)
    numerator = 0   #初始化分子
    denumerator = 0 #初始化分母
    for i in range(size):
        numerator += (x_train[i] - np.mean(x_train)) * (y_train[i] - np.mean(y_train))
        denumerator += (x_train[i] - np.mean(x_train)) ** 2

    w = numerator / denumerator
    b = np.mean(y_train) - w * np.mean(x_train)

    return w,b

#根据斜率w和截距b，输入x计算输出值
def predict(x,w,b):
    #预测模型
    y = w * x + b
    return y

# 根据W,B画图
# 测试集进行测试，并作图
def fit_line(w,b):
    #linspace 创建等差数列的函数    #numpy.limspace(start,stop,num,endpoint=True,retstep=False,dtype=None,axis=0#) 
    x = np.linspace(4,15,9)
    y = w*x+b
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    draw(x_train,y_train)
    w,b = fit(x_train,y_train)
    print(w,b)

fit_line(w,b)