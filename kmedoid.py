import random
class KMedoid:
    """
    A Class to do K-Medoid clustering
    """
    def __init__(self, distance_matrix, num_iters, num_clusters):
        self.__distance_matrix = distance_matrix;
        self.__size = len(distance_matrix);
        self.__num_iters = num_iters
        self.__result_clusters = []
        self.__num_clusters = num_clusters

    def process(self):
        self.__result_clusters = self.get_k_medoids_clusters(self.__distance_matrix, self.__num_iters, self.__num_clusters)

    def get_clusters(self):
        return self.__result_clusters;

    def get_k_medoids_clusters(self, distance_matrix, num_iters, num_clusters):
        # result = []
        ci = len(distance_matrix)
        # Algorithm description:
        # 1. Random pick medoids
        # 2. Update clusters
        # 3. Update medoids
        # 4. Repeat 2 and 3 for specified iterations
        medoids_ = random.sample(range(0, ci), num_clusters)
        clusters_ = []
        # Specified iterations by users
        for cur_iter in range(num_iters):
            clusters_ = self.update_cluster(distance_matrix, medoids_, ci)
            if (num_clusters > len(clusters_)):
                print("debug:", num_clusters, len(clusters_), len(medoids_))
                exit()
            medoids_ = self.update_medoids(distance_matrix, clusters_, ci)

        # result.append(clusters_)
        result = clusters_
        return result

    def update_cluster(self, distance_matrix, medoids_, ci):
        # select the nearest medoids for cur_p
        clusters_map = {}
        for med in medoids_:
            clusters_map[med] = []
        for p in range(ci):
            min_dist = -1.0
            select_med = -1
            for m in medoids_:
                if m == p:
                    select_med = m
                    break
                if min_dist < 0 or distance_matrix[m][p] < min_dist:
                    min_dist = distance_matrix[m][p]
                    select_med = m
            clusters_map[select_med].append(p)

        # clean up and return lists
        clusters_ = []
        for key, values in clusters_map.items():
            if len(values) > 0:
                clusters_.append(sorted(values))
            else:
                print("crazy:", key)
                for k in medoids_:
                    print(key, k, distance_matrix[key][k], distance_matrix[k][key])
                exit()


        if (len(clusters_) != len(medoids_)):
            print("not same:", len(clusters_), len(medoids_))
            exit()

        return clusters_

    def update_medoids(self, distance_matrix, clusters_, ci):
        # select the minimum distance to all others points in the same clusters
        medoids_ = []
        for cluster in clusters_:
            target_med = -1
            min_dist = -1.0
            for pi in cluster:
                cur_dist = 0.0
                for pj in cluster:
                    cur_dist += distance_matrix[pi][pj]
                if min_dist < 0 or cur_dist < min_dist:
                    min_dist = cur_dist
                    if pi < 0:
                        print("warning:", pi, pj, min_dist, cur_dist)
                    target_med = pi
            if target_med < 0:
                print("med error:", pi, target_med, min_dist, cluster)
                exit()

            medoids_.append(target_med)
        return medoids_

if __name__=='__main__':
    distance_matrix = [[0,1,2,3,4],[1,0,4,5,2],[2,4,0,2,3],[3,5,2,0,4],[4,2,3,4,0]];
    kmed = KMedoid(distance_matrix, 100, 2);
    kmed.process();
    print(kmed.get_clusters())
