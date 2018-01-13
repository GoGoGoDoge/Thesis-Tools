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


def get_hac_clusters(distance_matrix):
    result = []
    ci = len(distance_matrix)
    clusters = {}
    for i in range(ci):
        clusters[i] = []
        clusters[i].append(i)
    result.append(dict(clusters))
    X = ssd.squareform(distance_matrix)
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

def get_layer_clusters(result_clusters):
    result = []
    for layer in result_clusters:
        cur_cluster = []
        for label, cluster in layer.items():
            cur_cluster.append(cluster)
        if len(cur_cluster) > 1:
            result.insert(0,cur_cluster)
    return result

def cluster_score(distance_matrix, data_label, size):
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    (distance_matrix) = convert_2_square_matrix(distance_matrix,size)
    (clusterings) = get_hac_clusters(distance_matrix)
    (clusterings) = get_layer_clusters(clusterings)
    scores = []
    for clusters in clusterings:
        nClusters = len(clusters)
        nPos = 0.0
        nNeg = 0.0
        HY = 0
        for ip in range(0, size):
            if data_label[ip] == '+1':
                nPos = nPos + 1
            else:
                nNeg = nNeg + 1
        pPos = nPos/size
        pNeg = nNeg/size
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
            nPos = 0.0
            nNeg = 0.0
            for jc in range(0, nPoints):
                if data_label[clusters[ic][jc]] == "+1":
                    nPos = nPos + 1
                else:
                    nNeg = nNeg + 1
            pPos = nPos/nPoints
            pNeg = nNeg/nPoints
            pC = float(nPoints)/size

            if pPos > 0:
                H_Pos = -pPos*np.log2(pPos)
            if pNeg > 0:
                H_Neg = -pNeg*np.log2(pNeg)
            # Advise by Feng, remove negative sign
            HY_C[ic] = pC*(H_Pos + H_Neg)
            HC = HC - pC*np.log2(float(pC))
        #print(HC)
        #print(HY_C)
        IY_C = HY - sum(HY_C)
        #print(IY_C)
        score = 2*IY_C/(HY+HC)
        # print(nClusters, score)
        print(str(nClusters) + "," + str(score)+ "," + str(IY_C/HY)+ "," + str(IY_C/HC))
        scores.insert(0, score)

    return scores

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide file name! ")
        print("e.g. python3 parse_distance_hac_nmi.py your_distance.dm")
        exit()
    filename = sys.argv[1]
    (dist_mat, data_label, size) = parse_distance_matrix(filename)
    F = cluster_score(dist_mat, data_label, size)

    # Uncomment to check more information
    # plt.plot(F, 'r-')
    # plt.ylabel("NMI")
    # plt.xlabel("nclusters")
    # plt.show()
    #
    # print(F)
