import csv
def get_best_nmi(filename):
    try:
        with open(filename) as f:
            rows = csv.reader(f, delimiter=',')
            best_k = -1
            best_nmi = 0
            for row in rows:
                nmi = float(row[3])
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_k = row[2]
            return best_nmi, best_k
            print("Best NMI: ", best_nmi, " k: ", best_k)
    except:
        # print(filename, " not found!")
        return(0, -1)

if __name__ == '__main__':
    dataset = input("Enter dataset (colon, cystic, leukemia, moschitti, web): ")
    method = input("Enter clustering method (kmeans, hac, spec, kmedoid): ")
    kernels = ['ati', 'atr', 'cpi', 'cpr', 'spi', 'spr', 'ccti', 'cctr', 'sfi']
    # cystic_ati_kmeans_beta_tuned_d3.csv

    for degree in range(2, 11):
        best_nmi = 0
        best_k = -1
        best_kernel = ""
        for kernel in kernels:
            filename = str(dataset) + "_" + str(kernel) + "_" + method + "_beta_tuned_d" + str(degree) + ".csv"
            (nmi, k) = get_best_nmi(filename)
            if int(k) < 0:
                continue
            if nmi > best_nmi:
                best_nmi = nmi
                best_k = k
                best_kernel = kernel
        print(dataset, method, "best kernel:" + str(best_kernel), "degree:" + str(degree), "Best NMI: ", best_nmi, " when k: ", best_k)
