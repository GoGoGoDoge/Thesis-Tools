import os
import re
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt
from sklearn import metrics;
import random
import time
import matplotlib.pylab as plt
import sys

def LOG_COMPARE(before, after):
    print("Before: " + str(before).strip());
    print("After : " + str(after).strip());

def LOG_INFO(info):
    print("Info:" + str(info));

def insert_ast(s):
    # Inserts asterisks * for sympy: xy -> x*y, 3x -> 3*x
    p=re.compile(r'(\d)([xy])') # return a compiled pattern object
    res = re.sub(p, r'\1*\2',s)
    q=re.compile(r'xy')
    return re.sub(q, r'x*y', res) # pattern, replace, string

def fatal_error(gram_matrix_expression, index_1, index_2, exp, tokens):
    print("Fatal Error! ", (index_1, index_2), "has different value")
    print(gram_matrix_expression[(index_1, index_2)])
    print("vs")
    print(exp)
    print(tokens)
    exit()

def parse_distance_matrix(filename):
    # LOG_INFO(filename);
    head_p=re.compile(r'(\d+):(\S+)?') # ? represents greedy to match as much as possible
    # elmt_p=re.compile(r'(\d+):(\S+)')
    distance_matrix = {}
    labels = {}
    exp_row_idx = 0
    exp_col_idx = 0
    size = 0
    for line in open(filename, 'r'):
        tokens = insert_ast(line.rstrip()).split() # split by space
        match_head = re.match(head_p, tokens[0])
        if match_head:
            group = match_head.groups()
            if group[1] != None:
                i = int(group[0])
                size = max(size, i)
                labels[i] = group[1]
                # print("check labels:", i, labels[i])
            else:
                row_idx = int(group[0])
                for token in tokens[1:]:
                    match_exp = token.split(":")
                    col_idx = int(match_exp[0])
                    exp = match_exp[1]
                    index_1 = min(row_idx, col_idx)
                    index_2 = max(row_idx, col_idx)

                    if (index_1, index_2) in distance_matrix:
                        if distance_matrix[(index_1, index_2)] != float(exp):
                            fatal_error(distance_matrix, index_1, index_2, exp, tokens)
                    else:
                        distance_matrix[(index_1, index_2)] = float(exp)

    return (distance_matrix,labels, size + 1)

def convert_2_square_matrix(distance_matrix, size):
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(distance_matrix[min(i,j),max(i,j)])
        result.append(row)
    return result


def get_k_medoids_clusters(distance_matrix, num_iters):
    result = []
    ci = len(distance_matrix)
    # Algorithm description:
    # 1. Random pick medoids
    # 2. Update clusters
    # 3. Update medoids
    # 4. Repeat 2 and 3 for specified iterations
    for num_clusters in range (2, ci + 1): # shall add one here
        medoids_ = random.sample(range(0, ci), num_clusters)
        clusters_ = []
        # Specified iterations by users
        for cur_iter in range(num_iters):
            clusters_ = update_cluster(distance_matrix, medoids_, ci)
            if (num_clusters > len(clusters_)):
                print("debug:", num_clusters, len(clusters_), len(medoids_))
                exit()
            medoids_ = update_medoids(distance_matrix, clusters_, ci)

        result.append(clusters_)
    return result

def update_cluster(distance_matrix, medoids_, ci):
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

def update_medoids(distance_matrix, clusters_, ci):
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



def cluster_score(distance_matrix, data_label, data_size, num_iters = 10):
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    (distance_matrix) = convert_2_square_matrix(distance_matrix, data_size)
    (clusterings) = get_k_medoids_clusters(distance_matrix, num_iters)
    scores = []
    for clusters in clusterings:
        (NMI, IY_C, HY, HC) = get_NMI_score(data_label, clusters, data_size)
        IY_C_HY = float(IY_C)/HY
        IY_C_HC = float(IY_C)/HC
        (F) = get_fmeasure_score(data_label, clusters, data_size)
        (ARI) = get_rand_index(data_label, clusters, data_size)
        print(str(len(clusters)) + "," + str(NMI)+ "," + str(IY_C_HY)+ "," + str(IY_C_HC) + "," + str(F) + "," + str(ARI))
        scores.insert(0, F)

    return scores

def get_labels(clusters):
    total_elements = sum([len(cluster) for cluster in clusters]);
    sets = [set(cluster) for cluster in clusters];
    label_clusters = []
    for i in range(total_elements):
        for x,cur_set in enumerate(sets):
            if i in cur_set:
                label_clusters.append(x)
                break;
    return label_clusters

def get_rand_index(global_data_labels, clusters, data_size):
    cluster_true = [[],[]]
    for i in range(0, data_size):
        if global_data_labels[i] == "+1":
            cluster_true[0].append(i)
        else:
            cluster_true[1].append(i)
    if len(cluster_true[0]) == 0:
        cluster_true.remove([])

    label_clusters1 = get_labels(clusters)
    label_clusters2 = get_labels(cluster_true)
    score = metrics.adjusted_rand_score(label_clusters1, label_clusters2);

    return score;

def get_fmeasure_score(global_data_labels, clusters, data_size):
    nClusters = len(clusters)
    cluster_labels = [0 for x in range(nClusters)]
    confusion_mat = [[0,0],[0,0]]
    for ic in range(0, nClusters):
        # print("For the ith cluster: ", ic)
        nPoints = len(clusters[ic])
        # print("     Number of points in current cluster: ", nPoints)
        nPos = 0
        nNeg = 0
        for jc in range(0, nPoints):
            if global_data_labels[clusters[ic][jc]] == "+1":
                nPos = nPos + 1
            else:
                nNeg = nNeg + 1
        # print("For i th cluster, +1 v.s. -1 is: ", ic, nPos, nNeg)
        if nPos > nNeg:
            cluster_labels[ic] = 1 # pos cluster
            confusion_mat[1][1] = confusion_mat[1][1] + nPos # TP = confusion_mat[i][1][1]
            confusion_mat[1][0] = confusion_mat[1][0] + nNeg # FN = confusion_mat[i][1][0]
        else:
            cluster_labels[ic] = -1 # neg cluster
            confusion_mat[0][1] = confusion_mat[0][1] + nPos # FP = confusion_mat[i][0][1]
            confusion_mat[0][0] = confusion_mat[0][0] + nNeg # TN = confusion_mat[i][0][0]

    # then compute the score using the combined confusion matrix, e.g. use accuracy.
    TN = confusion_mat[0][0]
    FP = confusion_mat[0][1]
    FN = confusion_mat[1][0]
    TP = confusion_mat[1][1]
    # accuracy = (confusion_mat_sum[0][0]+confusion_mat_sum[1][1])/(confusion_mat_sum[0][0]+confusion_mat_sum[0][1]+confusion_mat_sum[1][0]+confusion_mat_sum[1][1])
    # print("Final accuracy for this set of parameter is: ", accuracy)
    # print("debug confusion: ", TN, FP, FN, TP)
    if FN+TP == 0:
        return 0
    TPR = TP/(FN+TP)
    if TP+FP == 0:
        return 0
    Precision = TP/(TP+FP)
    F = 2*(Precision*TPR)/(Precision+TPR)

    return F

def get_NMI_score(global_data_labels, clusters, data_size):
    nClusters = len(clusters)
    cluster_labels = [0 for x in range(nClusters)]

    nPos = 0
    nNeg = 0
    HY = 0
    for ip in range(0, data_size):
        if global_data_labels[ip] == '+1':
            nPos = nPos + 1
        else:
            nNeg = nNeg + 1
    pPos = nPos/data_size
    pNeg = nNeg/data_size
    H_Pos = 0
    H_Neg = 0
    if pPos > 0:
        H_Pos = -pPos*np.log2(pPos)
    if pNeg > 0:
        H_Neg = -pNeg*np.log2(pNeg)
    HY = H_Pos + H_Neg
    #print(HY)
    HY_C = [0 for i in range(nClusters)]
    HC = 0
    for ic in range(0, nClusters):
        nPoints = len(clusters[ic])

        nPos = 0
        nNeg = 0
        for jc in range(0, nPoints):
            if global_data_labels[clusters[ic][jc]] == "+1":
                nPos = nPos + 1
            else:
                nNeg = nNeg + 1
        pPos = nPos/nPoints
        pNeg = nNeg/nPoints
        pC = nPoints/data_size

        H_Pos = 0
        H_Neg = 0
        if pPos > 0:
            H_Pos = -pPos*np.log2(pPos)
        if pNeg > 0:
            H_Neg = -pNeg*np.log2(pNeg)

        HY_C[ic] = pC*(H_Pos + H_Neg)
        HC = HC - pC*np.log2(pC)

    IY_C = HY - sum(HY_C)
    NMI = 2*IY_C/(HY+HC)

    return NMI, IY_C, HY, HC

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide file name! ")
        print("e.g. python3 parse_distance_nmi.py your_distance.dm your_desired_iterations(default: 10)")
        exit()
    filename = sys.argv[1]
    (dist_mat, data_label, size) = parse_distance_matrix(filename)
    if len(sys.argv) > 2:
        score= cluster_score(dist_mat, data_label, size, int(sys.argv[2]))
    else:
        score= cluster_score(dist_mat, data_label, size)


    # Uncomment to check more information
    # plt.plot(score, 'r-')
    # plt.ylabel("NMI")
    # plt.xlabel("nclusters")
    # plt.show()
    #
    # print(score)
