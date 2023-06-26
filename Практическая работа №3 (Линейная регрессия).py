import pandas as pd
import numpy as np
import math
from math import sqrt
import operator
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data, col1, col2):
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.line_coef = None

    def linear_regression(self):
        '''Функция реализации алгоритма линейной регрессии'''
        X = np.asarray(self.data[self.col1].values.tolist())
        Y = np.asarray(self.data[self.col2].values.tolist())
        #X = X.reshape(len(X), 1)
        #Y = Y.reshape(len(Y), 1)
        n = len(X)
        M1 = np.array([[n, np.sum(X)], [np.sum(X), np.sum(X ** 2)]])
        v1 = np.array([np.sum(Y), np.sum(X * Y)])
        self.line_coef = np.linalg.solve(M1, v1)
        sq_error = 0
        m = 1
        for i in range(len(Y)):
            new_Y = self.line_coef[0] + self.line_coef[1] * X[i]
            sq_error += (new_Y - Y[i]) ** 2
        standard_error = sqrt(sq_error) / (n - m - 1)
        self.line_coef = np.append(self.line_coef, standard_error)
        print('Уравнение прямой: y = {0} + {1}x + {2}'.format(self.line_coef[0], self.line_coef[1], self.line_coef[2]))
        self.visualize(X, Y)

    def visualize(self, X, Y):
        '''Функция, отвечающая за визуализацию, где
        :param X: массив входных переменных,
        :param Y: массив выходных переменных,
        :param line_coef: массив, где первый элемент - b0, второй - b1, третий - е (стандартная ошибка)'''
        X = X.tolist()
        Y = Y.tolist()
        max_x = X.index(max(X))
        min_x = Y.index(min(Y))
        x = [X[min_x], X[max_x]]
        y = []
        for i in range(len(x)):
            y.append(self.line_coef[0] + self.line_coef[1] * x[i] + self.line_coef[2])
        plt.plot(x, y, color='#58b970', label='Линия регрессии')
        plt.scatter(X, Y, c='#ef5423', label='ID')
        plt.xlabel('Опыт работы')
        plt.ylabel('Зарплата')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('/Users/macbook/Desktop/6 семестр/Проектирование СППР/Программы/Salary_dataset.csv')
    col1 = 'YearsExperience'
    col2 = 'Salary'
    data = data.set_index('ID')
    lr = LinearRegression(data=data, col1=col1, col2=col2)
    lr.linear_regression()