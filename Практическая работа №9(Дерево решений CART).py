import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CART:
    def __init__(self):
        self.tree = None

    def split(self, x, y, feature, threshold):
        """Функция разделения выборки на две части"""
        left_idx = x[:, feature] < threshold
        right_idx = x[:, feature] >= threshold
        left = (x[left_idx], y[left_idx])
        right = (x[right_idx], y[right_idx])
        return left, right

    def find_gini(self, y):
        """Функция вычисления индекса Джини"""
        n_samples = y.shape[0]
        n_classes = len(np.unique(y))
        gini = 1.0
        for c in range(n_classes):
            gini -= (np.sum(y == c) / n_samples) ** 2
        return gini

    def most_common_label(self, y):
        """Функция поиска наиболее частого значения метки класса"""
        return np.bincount(y).argmax()

    def find_best_split(self, x, y):
        """Функция поиска наилучших параметров для разбиения
        :return best_feature, best_threshold: наилучший признак и наилучший порог разбиения"""
        best_feature, best_threshold, best_gain = None, None, 0
        for feature in range(x.shape[1]):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left, right = self.split(x, y, feature, threshold)
                if len(left[0]) == 0 or len(right[0]) == 0:
                    continue
                #расчет прироста информации
                gain = self.find_gini(y) - len(left[0]) / len(y) * self.find_gini(left[1]) - len(right[0]) / len(
                    y) * self.find_gini(right[1])
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        return best_feature, best_threshold

    def build_tree(self, x, y):
        """Функция, отвечающая за рекурсивное построение дерева,
        :return dict: словарь, который состоит из признака, порога, левой ветви и правой"""
        #Все объекты в листе относятся к одному классу
        if len(np.unique(y)) == 1:
            return self.most_common_label(y)
        best_feature, best_threshold = self.find_best_split(x, y)
        #Достигнута максимальная глубина
        if best_feature is None:
            return self.most_common_label(y)
        left, right = self.split(x, y, best_feature, best_threshold)
        return {'feature': best_feature, 'threshold': best_threshold, 'left': self.build_tree(left[0], left[1]),
                'right': self.build_tree(right[0], right[1])}

    def fit(self, x, y):
        """Функция обучения дерева"""
        self.tree = self.build_tree(x, y)

    def predict_sample(self, x, node):
        """Функция предсказания результата для одной строки данных,
        которая может вернуть сам узел, либо, если значение у выбранного признака превышает пороговое,
        то рекурсивно вызвать данный метод для левоц ветви, в противном случае - для правой"""
        if isinstance(node, dict):
            if x[node['feature']] < node['threshold']:
                return self.predict_sample(x, node['left'])
            else:
                return self.predict_sample(x, node['right'])
        else:
            return node

    def predict(self, x):
        """Функция предсказание для всех данных"""
        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            y_pred[i] = self.predict_sample(x[i], self.tree)
        return y_pred


if __name__=='__main__':
    # Загрузка набора данных
    data = load_iris()
    X = data.data
    y = data.target
    # Разделение набора данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Создание экземпляра модели CART
    model = CART()
    # Обучение модели на обучающих данных
    model.fit(X_train, y_train)
    # Предсказание целевой переменной на тестовых данных
    y_pred = model.predict(X_test)
    # вывод метрик качества модели
    print("Предсказанные метки классов: {}".format(y_pred))
    print("Действительные метки классов: {}".format(y_test))
    print("Accuracy:", accuracy_score(y_test, y_pred))
