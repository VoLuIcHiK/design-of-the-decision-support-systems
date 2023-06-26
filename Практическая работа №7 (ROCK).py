import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class ROCK:

    def __init__(self, eps, mu):
        self.eps = eps
        self.mu = mu
        self.cluster = None

    def dist(self, x1, x2):
        """Функция вычисления расстояние между двумя точками
        :param x1: координаты первой точки,
        :param x2: координаты второй точки"""
        # Функция для вычисления расстояния между двумя точками
        return np.linalg.norm(x1 - x2)


    def rock(self, data):
        """Основной алгоритм программы,
        :param data: массив данных для кластеризации"""
        n = len(data)
        #матрица связей
        links_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.dist(data[i], data[j])
                if d < self.eps:
                    links_matrix[i, j] = 1
                    links_matrix[j, i] = 1
        # Массив с числом связей для каждой точки
        num_links = np.sum(links_matrix, axis=1)
        # Массив с номером кластера для каждой точки
        cluster = np.zeros(n)
        # Номер текущего кластера
        cur_cluster = 1
        # Для каждой точки
        for i in range(n):
            if cluster[i] == 0:
                # Создаем новый кластер
                cluster[i] = cur_cluster
                # Находим все точки, связанные с текущей точкой
                neighbors = np.argwhere(links_matrix[i] == 1).flatten()
                # Для каждой связанной точки
                for j in neighbors:
                    if cluster[j] == 0:
                        # Добавляем ее в текущий кластер, если она удовлетворяет пороговому значению
                        if num_links[j] >= self.mu:
                            cluster[j] = cur_cluster
                # Увеличиваем номер текущего кластера
                cur_cluster += 1
        self.cluster = cluster

    def visualize(self, data):
        """Функция визуализации результата"""
        plt.scatter(data[:, 0], data[:, 1], c=self.cluster, cmap='rainbow')
        plt.show()



if __name__ == '__main__':
    # Сгенерируем данные
    np.random.seed(0)
    data = np.random.randn(100, 2)
    centers = [[2, 2], [-2, -1], [1, -2], [1, 1]]
    data, labels_true = make_blobs(n_samples=800, centers=centers, cluster_std=0.4,
                                random_state=0)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    # Кластеризуем данные с помощью ROCK
    model = ROCK(eps=0.5, mu=5)
    model.rock(data)
    # Визуализируем результаты
    model.visualize(data)
