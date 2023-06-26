import numpy as np  # Import the necessary packages.
import pandas as pd
from random import randint, sample, choice
from math import sqrt
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, col_1, col_2, data, k=None):
        self.col_1 = col_1
        self.col_2 = col_2
        self.data = data
        self.data_for_cluster = None
        self.clusters = None
        self.cluster_colors = None
        self.k = k

    def get_num_k(self):
        '''Задание числа кластеров k'''
        self.k = randint(2, 10)

    def choose_clusters_random(self):
        '''Выбор k кластеров записей, которые будут служить начальными центрами кластеров'''
        rnd = sample(range(0, len(data)), self.k)
        self.clusters = list()
        self.color = ["#" + ''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(self.k)]
        self.cluster_colors = dict(zip(rnd, self.color))
        for cluster in rnd:
            self.clusters.append(([data.at[cluster, self.col_1], data.at[cluster, self.col_2]], cluster))

    def find_distance(self, i, cluster):
        return sqrt((self.data.at[i, self.col_1] - cluster[0][0]) ** 2
                    + (self.data.at[i, self.col_2] - cluster[0][1]) ** 2)

    def create_df_for_cluster(self):
        columns = ['id', '№ cluster']
        for cluster in self.clusters:
            columns.append(cluster[1])
        self.data_for_cluster = pd.DataFrame(columns=columns)
        for i in range(len(data)):
            self.data_for_cluster.at[i, 'id'] = i
            for cluster in self.clusters:
                if cluster[1] == i:
                    self.data_for_cluster.at[i, cluster[1]] = 0.0
                else:
                    res = self.find_distance(i, cluster)
                    self.data_for_cluster.at[i, cluster[1]] = res
            buf = 999999
            for cluster in self.clusters:
                if self.data_for_cluster.at[i, cluster[1]] < buf:
                    self.data_for_cluster.at[i, '№ cluster'] = cluster[1]
                    buf = self.data_for_cluster.at[i, cluster[1]]

    def count_sum_squared_errors(self):
        sum = 0
        for i in range(len(data)):
            cluster = self.data_for_cluster.at[i, '№ cluster']
            sum += self.data_for_cluster.at[i, cluster] ** 2
        return sum

    def count_centroids(self):
        new_cluster_centroids = list()
        for cluster in self.clusters:
            new_x = 0.0
            new_y = 0.0
            l = 0
            for i in range(len(self.data_for_cluster)):
                if self.data_for_cluster.at[i, '№ cluster'] == cluster[1]:
                    new_x += self.data.at[i, self.col_1]
                    new_y += self.data.at[i, self.col_2]
                    l += 1
            new_cluster_centroids.append(([new_x/l, new_y/l], cluster[1]))
        self.clusters = new_cluster_centroids

    def show(self, iter):
        x = []
        y = []
        x_c = []
        y_c = []
        for i in range(len(self.data_for_cluster)):
            x.append(self.data.at[i, self.col_1])
            y.append(self.data.at[i, self.col_2])
            plt.scatter(x[i], y[i], color=self.cluster_colors[self.data_for_cluster.at[i, '№ cluster']])
            #plt.text(self.solution[val][0][0], self.solution[val][0][1], val)
        for i in range(len(self.clusters)):
            x_c.append(self.clusters[i][0][0])
            y_c.append(self.clusters[i][0][1])
            plt.scatter(x_c[i], y_c[i], color=self.cluster_colors[self.clusters[i][1]])
            plt.text(x_c[i], y_c[i], self.clusters[i][1])
        #plt.scatter(x, y)
        plt.title(f'Итерация №{iter}', fontsize=20, fontname='Times New Roman')
        plt.xlabel('X', color='gray')
        plt.ylabel('Y', color='gray')
        plt.show()

    def show_elbow(self, mas_sum_err):
        x = []
        y = []
        for i in range(len(mas_sum_err)):
            x.append(i + 1)
            y.append(mas_sum_err[i])
        plt.scatter(x, y)
        plt.title(f'Поиск оптимального значения кластеров', fontsize=20, fontname='Times New Roman')
        plt.xlabel('X', color='gray')
        plt.ylabel('Y', color='gray')
        plt.show()


    def main(self):
        # k = get_num_k()
        iterations = 5
        mas_sum_err = list()
        buf = self.data.copy()
        for k in range(1, 11):
            self.data = buf.copy()
            self.k = k
            self.choose_clusters_random()
            self.create_df_for_cluster()
            self.count_centroids()
            mas_sum_err.append(self.count_sum_squared_errors())
        self.show_elbow(mas_sum_err)
        self.k = 3
        self.data = buf.copy()
        self.choose_clusters_random()
        for i in range(1, iterations + 1):
            self.create_df_for_cluster()
            self.count_centroids()
            mas_sum_err.append(self.count_sum_squared_errors())
            self.show(i)


if __name__ == '__main__':
    data = pd.read_csv('/Users/macbook/Desktop/6 семестр/Проектирование СППР/Программы/Mall_Customers.csv')
    akg = Kmeans(data=data, col_1='Annual_Income_(k$)', col_2='Spending_Score_(1-100)')
    akg.main()