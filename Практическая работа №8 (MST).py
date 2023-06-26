import numpy as np
import heapq
from sklearn.datasets import load_wine
from collections import deque
import networkx as nx
from scipy.spatial.distance import pdist

class MST:

    def __init__(self, data, k):
        self.data = data
        self.graph = None #матрица смежности графа
        self.visited = None #массив для хранения посещенных вершин
        self.parent = None #массив для хранения родительских вершин остовного дерева
        self.distance = None #массив для хранения расстояний от текущей вершины до других
        self.heap = [(0, 0)] #куча для хранения необработанных вершин графа
        self.n = len(self.data)
        self.k = k #кол-во кластеров
        self.min_st = None

    def find_dist(self, x1, x2):
        """Функция вычисления расстояние между двумя объектами
        :param x1: массив данных первого объекта,
        :param x2: массив данных второго объекта"""
        # Функция для вычисления расстояния между двумя точками
        return np.linalg.norm(x1 - x2)

    def create_graph(self):
        """Функция создания графа из набора данных"""
        self.graph = np.zeros(shape=(self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.graph[i][j] = 0.0
                else:
                    if self.graph[i][j] == 0:
                        self.graph[i][j] = self.find_dist(self.data[i], self.data[j])
                        self.graph[j][i] = self.find_dist(self.data[j], self.data[i])
        print('Граф, построенный по начальным данным')
        print(self.graph)

    def find_max_edge(self, mas):
        """Функция поиска ребер с максимальной стоимостью"""
        max = -99
        cord = (-99, -99)
        for i in range(len(mas)):
            if self.min_st[i, mas[i]] > max:
                max = self.min_st[i, mas[i]]
                cord = (i, mas[i])
        return cord[0], cord[1]

    def make_clusters(self):
        for k in range(self.k - 1):
            mas = np.argmax(self.min_st, axis=1)
            i, j = self.find_max_edge(mas)
            self.min_st[i, j] = 0.0
            self.min_st[j, i] = 0.0
        print('После удаления ребер')
        print(self.min_st)

    def mst(self):
        """Функция, реализующая основной алгоритм программы"""
        self.create_graph()
        n = self.graph.shape[0]
        self.visited = np.zeros(n, dtype=bool)
        self.parent = np.zeros(n, dtype=int) - 1
        self.dist = np.full(n, np.inf)
        self.dist[0] = 0
        while self.heap:
            #Извлекается вершина с минимальным расстоянием из кучи.
            (dist, node) = heapq.heappop(self.heap)
            #Проверка была ли посещена вершина
            if self.visited[node]:
                continue
            #вершина отмечается как посещенная
            self.visited[node] = True
            for i in range(n):
                '''Если расстояние до соседа меньше текущего расстояни до этого соседа, 
                то обновляется информация о родительской вершине, расстоянии до соседа,
                сосед добавляется в кучу'''
                if self.graph[node, i] != 0 and not self.visited[i] and self.graph[node, i] < self.dist[i]:
                    self.dist[i] = self.graph[node, i]
                    self.parent[i] = node
                    heapq.heappush(self.heap, (self.dist[i], i))
        #Сбор результата
        res = np.zeros((n, n))
        for i in range(1, n):
            res[i, self.parent[i]] = self.dist[i].round(2)
            res[self.parent[i], i] = self.dist[i].round(2)
        print("Матрица смежности минимального остовного дерева для данного графа:")
        print(res)
        self.min_st = res
        self.make_clusters()
        return res


if __name__=='__main__':
    data = load_wine()
    # разделение на тренировочную и тестовую выборки
    X = data.data[:8,:]
    y = data.target
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MST(data=X, k=len(np.unique(y)))
    res = model.mst()
