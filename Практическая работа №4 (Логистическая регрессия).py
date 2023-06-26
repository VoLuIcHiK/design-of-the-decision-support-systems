import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.1, num_iter=1000):
        self.X = X
        self.y = y.to_numpy().reshape(-1, 1)
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.num_samples, self.num_features = X.shape
        #числовая экспоненциальная стабильность
        self.eps = 1e-6
        self.weight = None
        self.bias = None

    def loss(self, y, y_pred):
        '''Функция потерь'''
        return -np.mean(
            (y * np.log(y_pred + self.eps) - (1. - y) * np.log(1. - y_pred + self.eps)),
            axis=1
        )

    def activation(self, z):
        '''Функция активации - сигмоида'''
        return 1 / (1 + np.exp(-z))

    def weight_and_bias(self):
        '''Функция, отвечающая за начальную иниициализацию весов и биаса'''
        self.weight = np.zeros((self.num_features, 1))
        self.bias = 0

    def fit(self):
        '''Функция, отвечающая за процесс обучения модели, где
        :param X: матрица данных,
        :param y: матрица целевого столбца (который надо предсказать)'''
        self.weight_and_bias()
        for i in range(self.num_iter):
            #предсказание значения целевого столбца
            w_x = np.dot(self.X, self.weight) + self.bias
            z = self.activation(w_x)
            #вычисление градиентов
            dw = (1 / self.num_samples) * np.dot(self.X.T, (z - self.y))
            db = (1 / self.num_samples) * np.sum(z - self.y)
            #обновление значений весов и биаса
            dw = dw.reshape(self.weight.shape)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            print("Epoch: {0}; loss: {1}".format(i, self.loss(self.y, z).mean()))

    def predict(self, X):
        '''Функция, отвечающая за предсказание значений целевого столбца, где
        :param X: матрица данных'''
        w_x = np.dot(X, self.weight) + self.bias
        z = self.activation(w_x)
        y_pred = np.round(z)
        return y_pred


if __name__ == '__main__':
    data = pd.read_csv("heart_diseases.csv")
    #удаление NaN из данных
    data = data.fillna(data.mean())
    #разделение на тренировочную и тестовую выборки
    X = data.drop(["TenYearCHD"], axis=1)
    y = data["TenYearCHD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #создание экземпляра модели
    model = LogisticRegression(X=X_train, y=y_train)
    #обучение модели
    model.fit()
    #предсказание меток
    y_pred = model.predict(X_test)
    #вывод метрик качества модели
    print("Accuracy:", accuracy_score(y_test, y_pred))



