"""
I write this function to parse gram matrix and reconstruct the program for parsing GP
"""
import os
import re
import sympy
import numpy as np
import sys
from hac_ch import HAC_CH

# --- Global Variables Declaration --- #
global_gram_expression = {}
global_data_labels = {}
global_size = 0
clusters = []
cluster_labels = []
global_filename = ""
global global_nmi
global global_I_HC1 # I/H_Y
global global_I_HC2 # I/H_C

def LOG_INFO(info):
    print("Info:" + str(info));

def LOG_COMPARE(before, after):
    print("Before: " + str(before).strip());
    print("After : " + str(after).strip());

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

def parse_gram_matrix(filename, is_sympy = False):
    head_p=re.compile(r'(\d+):(\S+)?') # ? represents greedy to match as much as possible
    # elmt_p=re.compile(r'(\d+):(\S+)')
    gram_matrix_expression = {}
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

                    if is_sympy:
                        exp = sympy.sympify(exp)

                    if (index_1, index_2) in gram_matrix_expression:
                        if gram_matrix_expression[(index_1, index_2)] != exp:
                            fatal_error(gram_matrix_expression, index_1, index_2, exp, tokens)
                    else:
                        gram_matrix_expression[(index_1, index_2)] = exp

    return (gram_matrix_expression,labels, size + 1)

def get_gram_vals (gram_exp, size, alpha, beta, is_sympy = False):
    row_size = size
    col_size = size
    gram_value = np.arange(row_size*col_size,dtype=np.float).reshape((row_size,col_size))
    for i in range(row_size):
        for j in range(col_size):
            index = (min(i,j), max(i,j))
            expression = gram_exp[index]
            value = 0.0
            if is_sympy:
                value = gram_exp[index].subs([('x', alpha), ('y', beta)])
            else:
                value_exp = re.sub(r'x', str(alpha), expression)
                value_exp = re.sub(r'y', str(beta), value_exp)
                value_exp = re.sub(r'\^', r'**', value_exp)
                value = eval(value_exp)
            if isNaN(value):
                print("Fatal Error: ", i, j, value);
                exit()
            gram_value[i,j] = value

    return gram_value

def output2file(gram_value, filename):
    np.savetxt(filename, gram_value)

# Validate the expression from two matrix
def validate_exp(mat1, size1, mat2, size2):
    if size1 != size2:
        print ("Error: two arrays have different size", size1, "vs", size2)
        exit()
    row_size = size1
    col_size = size2
    if len(mat1) != len(mat2):
        print ("Error: two arrays have different size", len(mat1), "vs", len(mat2))
        exit()
    for i in range(row_size):
        for j in range(col_size):
            index = (min(i, j), max(i, j))
            s1 = str(mat1[index])
            s1 = re.sub(r'\^', r'**', s1)
            s1 = s1.replace(" ", "")

            s2 = str(mat2[index])
            s2 = s2.replace(" ", "")

            if s1 != s2:
                print("Error: different expression at pos:", i, j)
                LOG_COMPARE(s1, s2)
                exit()
    print("Mat1 and Mat2 are the same with the size: " + str(size1))

def validate_vals(mat1, size1, mat2, size2):
    if size1 != size2:
        print ("Error[vals]: two arrays have different size", size1, "vs", size2)
        exit()
    row_size = size1
    col_size = size2
    if len(mat1) != len(mat2):
        print ("Error[vals]: two arrays have different size", len(mat1), "vs", len(mat2))
        exit()
    for i in range(row_size):
        for j in range(col_size):
            if abs(mat1[i,j] - mat2[i,j]) > 0.000001:
                print("Error: two matrix values are not close")
                print(mat1[i,j], "vs", mat2[i,j])
                exit();
    print("Mat1 and Mat2 are the same in value with the size: " + str(size1))

def check_none(mat, size):
    row_size = size
    col_size = size
    for i in range(row_size):
        for j in range(col_size):
            index = (min(i, j), max(i, j))
            if index not in mat:
                print("Fatal Error: missing element", i, j);
                exit()

def isNaN(num):
    return num != num

def is_symmetric(mat):
    if len(mat) == 0:
        return False, 0, 0;
    N = len(mat);
    M = len(mat[0]);
    if N != M:
        return False, N, M;
    for i in range(N):
        for j in range(i, M):
            if mat[i][j] != mat[j][i]:
                print ("error diff between", i, j, mat[i][j], mat[j][i]);
                return False, N, M;
    return True, N, M;

def innerP2distance(_gm, _size):
    _dm = [[0 for aa in range(_size)] for bb in range(_size)]
    for i in range(_size):
        for j in range(_size):
            value = np.sqrt(_gm[i][i] + _gm[j][j] - 2 * _gm[i][j])
            if value != value:
                print("Fatal Error(not a number): ", i, j, value, _gm[i][i], _gm[j][j], _gm[i][j], _gm[j][i], np.sqrt(_gm[i][i] + _gm[j][j] - 2 * _gm[i][j]))
                return None
            _dm[i][j] = value

    return _dm

def cluster_score(alpha=1., beta=0.): # change to NMI!
    '''
    # Performs cross validation on a gram matrix of training data and
    # returns the averaged accuracy scores.
    # The gram matrix 'gm' is generated from 'get_elmnt'.
    # The number of folds is specified by the variable 'cv'.
    '''
    # print("This is neg_cv_score, alpha = ", alpha, "beta = ", beta)
    numpy_gm = get_gram_vals(global_gram_expression, global_size, alpha, beta)
    gm = numpy_gm.tolist()
    dm = innerP2distance(gm, global_size) # this is the pairwise distance matrix
    if dm == None:
        print("Invalid dm obtained!")
        return -999.0,0,0,0
    confusion_mat = [[0,0],[0,0]]
    (symmetric, row, col) = is_symmetric(dm);
    if symmetric == False:
        print("distance matrix:", row, col)
        print(dm)
        print("alpha:", alpha, "beta:", beta);
        return -999.0,0,0,0;
    # print("dm pass symmetric test")

    (symmetric, row, col) = is_symmetric(gm);
    if symmetric == False:
        print("gram matrix:", row, col)
        print(gm)
        print("alpha:", alpha, "beta:", beta);
        return -999.0,0,0,0;
    # print("gm pass symmetric test")
    global register_n_cluster
    hac_instance = HAC_CH(dm, gm)
    hac_instance.process_with_k(register_n_cluster)
    # print("hac complete...")

    global clusters
    global cluster_labels

    clusters = hac_instance.get_clusters()
    nClusters = len(clusters)
    cluster_labels = [0 for x in range(nClusters)]

    nPos = 0
    nNeg = 0
    HY = 0
    for ip in range(0, global_size):
        if global_data_labels[ip] == '+1':
            nPos = nPos + 1
        else:
            nNeg = nNeg + 1
    pPos = nPos/global_size
    pNeg = nNeg/global_size
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
        pC = nPoints/global_size

        H_Pos = 0
        H_Neg = 0
        if pPos > 0:
            H_Pos = -pPos*np.log2(pPos)
        if pNeg > 0:
            H_Neg = -pNeg*np.log2(pNeg)

        HY_C[ic] = pC*(H_Pos + H_Neg)
        HC = HC - pC*np.log2(pC)
    #print(HC)
    #print(HY_C)
    IY_C = HY - sum(HY_C)
    #print(IY_C)
    NMI = 2*IY_C/(HY+HC)
    global global_nmi
    global_nmi = NMI
    global global_I_HC1 # I/H_Y
    global_I_HC1 = IY_C/HY;
    global global_I_HC2
    global_I_HC2 = IY_C/HC
    #print('===============================')
    #print(str(nClusters) + "," + str(NMI)+ "," + str(IY_C/HY)+ "," + str(IY_C/HC))
    return NMI, IY_C, HY, HC

def save2file(F, alpha, beta, nClusters, ch_index, index_dict):
    print("F-measure:", F, "alpha:", alpha, "beta", beta, "# of clusters", nClusters, "CH-index:", ch_index)
    global global_filename
    print("filename:", global_filename)
    filename = global_filename.split(".")[0];
    try:
        os.stat(filename)
    except:
        os.mkdir(filename)
    assistFile(F, filename + "//" + filename + ".fmeasure")
    assistFile(alpha, filename + "//" + filename + ".alpha")
    assistFile(beta, filename + "//" + filename + ".beta")
    assistFile(nClusters, filename + "//" + filename + ".nclusters")
    assistFile(ch_index, filename + "//" + filename + ".ch_index")
    #Store the list of ch for ploting and study
    l = sorted(index_dict.items())
    assistWriteListFile(l, filename + "//" + filename + ".iter_ch_index")

def assistWriteListFile(info, filename):
    fe = open(filename, "a+")
    for tuple in info:
        fe.write(str(tuple[0]) + " " + str(tuple[1]) + "\r\n")
    fe.write("---###End###---\r\n")
    fe.close()


def assistFile(info, filename):
    fe = open(filename, "a+")
    fe.write(str(info) + "\r\n")
    fe.close()

def register(gram_exp, data_label, size, filename):
    global global_gram_expression
    global global_data_labels
    global global_size
    global global_filename
    global_gram_expression = gram_exp
    global_data_labels = data_label
    global_size = size
    global_filename = filename

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide file name! ")
        print("e.g. python3 kernel_grid_search_hac_nmi_tune.py your_kernel.kernel your_desired_step_size(default: 0.01)")
        exit()
    filename = sys.argv[1]
    step_size = 0.01
    if len(sys.argv) > 2:
        if float(sys.argv[2]) < 0:
            print("Step size cannot be negative number!")
            exit()
        step_size = float(sys.argv[2])

    # --- My Parse Version (eval) --- #
    (gram_exp, data_label, size1) = parse_gram_matrix(filename)
    #check_none(gram_exp, size1) # check if any expression not in the gram matrix
    # gram_vals = get_gram_vals(gram_exp, size1, 0.682164592364,0.394701288307)
    # dist_vals = innerP2distance(gram_vals, size1)
    #output2file(gram_vals, 'web_gram.txt')

    # --- Gram Sympy Version --- #
    # (gram_exp_sympy, data_label, size2) = parse_gram_matrix(filename, True)
    # gram_vals_sympy = get_gram_vals(gram_exp_sympy, size2, 0.682164592364,0.394701288307, True)
    #output2file(gram_vals_sympy, 'web_gram_sympy.txt')

    # --- Validate Part --- #
    # validate(gram_exp, size1, gram_exp_sympy, size2)
    # validate_vals(gram_vals, size1, gram_vals_sympy, size2)


    # --- Register globally --- #
    register(gram_exp, data_label, size1, filename)

    for decided_n_cluster in range(2, size1+1):
        global register_n_cluster
        register_n_cluster = decided_n_cluster
        # Grid Search
        alpha = 0.0
        beta = 0.0
        score = -998.0
        target_alpha = -1.0
        target_beta = -1.0

        while alpha < 1.0:
            while beta < 1.0:
                if beta >= alpha:
                    break
                (cur_score, IY_C, HY, HC) = cluster_score(alpha, beta)
                if cur_score > score:
                    score = cur_score
                    target_alpha = alpha
                    target_beta = beta
                beta += step_size
            alpha += step_size

        print(str(register_n_cluster) + "," + str(score)+ "," + str(IY_C/HY)+ "," + str(IY_C/HC))