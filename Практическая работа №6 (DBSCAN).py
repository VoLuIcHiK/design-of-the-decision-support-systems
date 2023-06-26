import numpy as np
from sklearn.datasets import make_blobs
import numpy.random as random
from  numpy.core.fromnumeric import *
import matplotlib.pyplot as plt

class DBSCAN:

    def __init__(self, X, e, minimum_points):
        self.e = e
        self.minimum_points = minimum_points
        self.X = X

    def find_neighbors(self, data_point):
        """Функция, отвечающая за поиск соседей, входящих в окрестность точки"""
        neighbors = []
        for idx, d in enumerate(self.X):
            distance = np.linalg.norm(data_point - d)
            if distance <= self.e and distance != 0:
                neighbors.append(idx)
        return neighbors

    def dbscan(self):
        """Функция, отвечающая за реализацию алгоритма DBSCAN"""
        labels = np.zeros(self.X.shape[0])
        cluster = 0
        for idx, data_point in enumerate(self.X):
            #если у этой точки уже имеется метка - пропускаем ее
            if labels[idx] != 0:
                continue
            #поиск соседей в округе
            neighbors = self.find_neighbors(data_point)
            #если кол-во соседей превышает пороговое
            if len(neighbors) >= self.minimum_points:
                cluster += 1
                labels[idx] = cluster
                for neighbor in neighbors:
                    if labels[neighbor] != 0:
                        continue
                    labels[neighbor] = cluster
                    #включение граничных точек в кластер
                    border_neighbor = self.find_neighbors(self.X[neighbor])
                    if len(border_neighbor) >= self.minimum_points:
                        neighbors += border_neighbor
        return labels

if __name__ == '__main__':
    centers = [[2, 2], [-2, -1], [1, -2], [1, 1]]
    X, labels_true = make_blobs(n_samples=800, centers=centers, cluster_std=0.4,
                                random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    model = DBSCAN(X=X, e=0.3, minimum_points=15)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=14, c=model.dbscan(), cmap='rainbow')
    plt.show()