from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt
from sklearn import metrics;
import random
import time

class HAC_CH:
    """
    A class created to do clustering
    """
    def __init__(self, distance_matrix, gram_matrix):
        self.__distance_matrix = distance_matrix;
        self.__size = len(distance_matrix);
        self.__gram_matrix = gram_matrix;
        self.__total_number_data =len(gram_matrix);
        self.__last_average_distance =[[0] for i in range(len(distance_matrix))];
        # All calculation is based on index of cluster
        self.__clusters = [[i] for i in range(len(distance_matrix))];

        # Result Clusters
        self.__result_cluster = [];
        self.__CH_INDEX = -999;
        self.__result_labels = [];
        self.__index_each_iter = {};

    def get_clusters(self):
        return self.__result_cluster;

    def get_index(self):
        return self.__CH_INDEX;

    def get_dict_index(self):
        return self.__index_each_iter;

    def get_label(self):
        labels = []
        visited = {}
        for i in range(self.__size):
            if self.__result_labels[i] in visited:
                labels.append(visited[self.__result_labels[i]])
            else:
                visited[self.__result_labels[i]] = i;
                labels.append(i)
        return labels;

    def process(self):
        # Initialize
        # print("hac stage 1")
        (self.__result_clusters) = self.__get_hac_clusters(self.__distance_matrix)
        # print("hac stage 2")
        (self.__labels) = self.__get_labels(self.__result_clusters)
        # print("hac stage 3")
        (self.__total_last_clusters) = self.__get_layer_clusters(self.__result_clusters)
        for idx in range(80, min(200, len(self.__total_last_clusters))): # This will provide the most detailed process
        # for idx in random.sample(range(50, min(150,len(self.__total_last_clusters))), 10): # This will random pick sample
            # print("hac stage 4: ", idx)
            last_clusters = self.__total_last_clusters[self.__size - idx - 1]
            # print("debug:", self.__size, idx,  self.__size - idx, len(last_clusters))
            if len(last_clusters) > 2 and len(last_clusters) < self.__size:
                # start_time = time.time();
                # ch_index = self.__calculate_Calinski_Harabasz_index(last_clusters);
                # print("Old time:", time.time() - start_time)
                # start_time = time.time();
                ch_index = self.__optimized_calculate_Calinski_Harabasz_index(last_clusters);
                self.__index_each_iter[idx] = ch_index # Store into a map for later ploting
                # print("New time:", time.time() - start_time)
                # print("Size:", len(last_clusters), "Compare:", ch_index, optimize_ch_index)
                if ch_index > self.__CH_INDEX:
                    # print("best size updated: ", idx, len(last_clusters))
                    self.__result_cluster = last_clusters
                    self.__CH_INDEX = ch_index
                    self.__result_labels = self.__labels[idx]

    def process_with_k(self, k):
        # Initialize
        # print("hac stage 1")
        (self.__result_clusters) = self.__get_hac_clusters(self.__distance_matrix)
        # print("hac stage 2")
        (self.__labels) = self.__get_labels(self.__result_clusters)
        # print("hac stage 3")
        (self.__total_last_clusters) = self.__get_layer_clusters(self.__result_clusters)
        self.__result_cluster = self.__total_last_clusters[self.__size - k]


    def __get_hac_clusters(self, X):
        result = []
        ci = len(X)
        clusters = {}
        for i in range(ci):
            clusters[i] = []
            clusters[i].append(i)
        result.append(dict(clusters))
        X = ssd.squareform(X)
        # Other available method: [single, complete, average, weighted, centroid, median, ward]
        Z = linkage(X, 'complete')
        idx = 1
        for layer in Z:
            i1 = int(layer[0])
            i2 = int(layer[1])
            clusters[ci] = clusters[i1] + clusters[i2]
            clusters.pop(i1)
            clusters.pop(i2)
            result.append(dict(clusters))
            ci += 1
        return result

    def __get_labels(self, result_clusters):
        result = []
        for layer in result_clusters:
            # print layer
            labels = [0 for i in range(self.__size)]
            for label, cluster in layer.items():
                # print label, cluster
                for p in cluster:
                    labels[p] = label
            result.append(labels)
        return result

    def __get_layer_clusters(self, result_clusters):
        result = []
        for layer in result_clusters:
            cur_cluster = []
            for label, cluster in layer.items():
                cur_cluster.append(cluster)
            result.append(cur_cluster)
        return result

    def __get_average_distance(self, cluster_a, cluster_b):
        """
        Calculate the average distance after combination of two clusters
        """
        result_cluster = cluster_a + cluster_b;
        count = 0;
        distance = 0.0;
        for i in range(len(result_cluster)):
            for j in range(i + 1, len(result_cluster)):
                count += 1;
                distance += self.__distance_matrix[result_cluster[i]][result_cluster[j]];
        return distance / count;

    def __distance_sqrt(self, index_object, cluster):
        n = len(cluster)
        ans = 0.0
        for i in cluster:
            ans += self.__distance_matrix[index_object][i]
        return ans / n / n

    def __calculate_Calinski_Harabasz_index(self, clusters):
        # Calculate the CH index
        number_of_cluster = len(clusters);
        numerator = 0.0
        denominator = 0.0
        for i in range(number_of_cluster):
            # Build a set to contains current cluster element
            cur_cluster = set(clusters[i]);
            numerator += len(cur_cluster) * self.__cluster_centroid_to_dataset_centroid(cur_cluster)
        numerator = numerator / (number_of_cluster - 1)

        for i in range(number_of_cluster):
            # Build a set to contains current cluster element
            cur_cluster = set(clusters[i]);
            denominator += self.__cluster_avg_square_distance(cur_cluster);
        denominator = denominator / (len(self.__gram_matrix) - number_of_cluster);
        denominator = max(denominator, 0.0000001)
        # print("old", len(clusters), numerator, denominator)
        return numerator / denominator;

    def __optimized_calculate_Calinski_Harabasz_index(self, clusters):
        # I optimize the calculation
        NC = len(clusters);
        intra_cluster_dist = 0.0
        inter_cluster_dist = 0.0
        self_dist = 0.0

        for cluster in clusters:
            ni = len(cluster)
            sum_intra_cluster_dist = 0.0
            for xi in cluster:
                for xj in cluster:
                    sum_intra_cluster_dist = sum_intra_cluster_dist + self.__gram_matrix[xi][xj]
            intra_cluster_dist = intra_cluster_dist + sum_intra_cluster_dist / ni

        for yi in range(self.__size):
            self_dist = self_dist + self.__gram_matrix[yi][yi]
            for yj in range(self.__size):
                inter_cluster_dist = inter_cluster_dist + self.__gram_matrix[yi][yj]

        inter_cluster_dist = inter_cluster_dist / self.__size;

        numerator = (intra_cluster_dist - inter_cluster_dist) * ((self.__size - NC) / (NC - 1))
        denominator = (self_dist - intra_cluster_dist)
        # print("new", len(clusters), numerator / denominator)
        if denominator < 0.001:
            return 0.0
        return numerator / denominator;

    def __cluster_avg_square_distance(self, cur_cluster):
        res = 0.0
        center_i = 0.0
        for i in cur_cluster:
            for j in cur_cluster:
                center_i += self.__gram_matrix[i][j]
        center_i = center_i / len(cur_cluster) / len(cur_cluster)

        for x in cur_cluster:
            A = self.__gram_matrix[x][x];
            B = 0.0
            for i in cur_cluster:
                B += self.__gram_matrix[x][i];
            res += (A - B * 2 / len(cur_cluster) + center_i)

        return res;

    def __cluster_centroid_to_dataset_centroid(self, cur_cluster):
        other_points = set(range(len(self.__gram_matrix))) - cur_cluster;
        n = len(cur_cluster)
        m = len(self.__gram_matrix)
        A = 0.0
        B = 0.0
        C = 0.0
        for i in cur_cluster:
            for j in cur_cluster:
                A += self.__gram_matrix[i][j];
        A = A * (m - n) * (m - n);

        for i in cur_cluster:
            for j in other_points:
                B += self.__gram_matrix[i][j];
        B = 2 * (m - n) * n * B;

        for i in other_points:
            for j in other_points:
                C += self.__gram_matrix[i][j];
        C = n * n * C;
        return (A - B + C) / n / n / m / m;


if __name__=='__main__':
    distance_matrix = [[0,1,2,3,4],[1,0,4,5,2],[2,4,0,2,3],[3,5,2,0,4],[4,2,3,4,0]];
    gram_matrix = [[1,2,3,4,5],[2,3,4,5,6],[4,5,6,7,8],[9,8,7,6,5],[8,7,6,5,4]];
    hac = HAC_CH(distance_matrix, gram_matrix);
    hac.process();
    print(hac.get_clusters())
    print(hac.get_label())
