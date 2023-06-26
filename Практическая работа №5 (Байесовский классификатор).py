import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

class NaiveBayes:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_samples, self.num_features = X.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        #числовая экспоненциальная стабильность
        self.eps = 1e-6

    def fit(self):
        #начальная инициализацияя среднего
        self.class_mean = np.zeros((self.num_classes, self.num_features), dtype=np.float64)
        #начальная инициализацияя стандартного отклонения
        self.class_var = np.zeros((self.num_classes, self.num_features), dtype=np.float64)
        #начальная инициализацияя предварительной вероятности для каждого класса
        self.class_priors = np.zeros(self.num_classes, dtype=np.float64)
        for c in self.classes:
            #выделение строк данных определенного класса
            X_c = self.X[self.y == c]
            self.class_mean[c, :] = X_c.mean(axis=0)
            self.class_var[c, :] = X_c.var(axis=0)
            self.class_priors[c] = X_c.shape[0] / float(self.num_samples)

    def gauss_density_func(self, X, class_id):
        """Закон распределения Гаусса (функция плотности)"""
        #числитель
        numerator = np.exp(- (X - self.class_mean[class_id]) ** 2 / (2 * self.class_var[class_id]))
        #знаменатель
        denominator = np.sqrt(2 * np.pi * self.class_var[class_id])
        return numerator / denominator

    def predict(self, X):
        # Рассчитываем вероятности для каждого класса
        #posteriors = []
        predictions = np.zeros((len(X), self.num_classes))
        for c in range(self.num_classes):
            prior = np.log(self.class_priors[c])
            prediction = np.sum(np.log(self.gauss_density_func(X, c)), axis=1)
            prediction = prior + prediction
            #posteriors.append(posterior)
            predictions[:, c] = prediction
        # Возвращаем класс с наибольшей вероятностью
        return np.argmax(predictions, 1)


if __name__ == '__main__':
    data = load_iris()
    # разделение на тренировочную и тестовую выборки
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Обучаем классификатор
    model = NaiveBayes(X=X_train, y=y_train)
    model.fit()
    # Предсказываем метки классов для тестовой выборки
    y_pred = model.predict(X_test)
    #вывод метрик качества модели
    print("Предсказанные метки классов: {}".format(y_pred))
    print("Действительные метки классов: {}".format(y_test))
    print("Accuracy:", accuracy_score(y_test, y_pred))



