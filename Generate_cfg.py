import csv

""" e.g. configure file
kernel: /Users/marco/Documents/CMU_Course/Thesis/Kernel/a_t_i_o_leukemia.kernel
alpha_min: 0.1
alpha_max: 2.0
alpha_div: 20
beta_min: 0.0
beta_max: 2.0
beta_div: 21
k_min: 2
k_max: 50
k_div: 49
degree_max: 1
link: ward

"""

if __name__ == '__main__':
    datasets = ['colon','cystic','leukemia','moschitti','web']
    kernels = ['a_t_i_o_', 'a_t_r_o_', 'c_p_i_o_', 'c_p_r_o_', 's_p_i_o_', 's_p_r_o_', 'c_ct_i_o_', 'c_ct_r_o_', 's_f_i_o_']
    # cystic_ati_kmeans_beta_tuned_d3.csv

    # Generate configuration files
    for dataset in datasets:
        for kernel in kernels:
            kernel_file = kernel + dataset + ".kernel"
            config_file = kernel + dataset + ".cfg"
            with open(config_file,'w') as f:
                f.write("kernel: /Users/marco/Documents/CMU_Course/Thesis/Kernel/" + str(kernel_file) + "\n")
                f.write("alpha_min: 0.1\n")
                f.write("alpha_max: 2.0\n")
                f.write("alpha_div: 20\n")
                f.write("beta_min: 0.0\n")
                f.write("beta_max: 2.0\n")
                f.write("beta_div: 21\n")
                f.write("k_min: 2\n")
                f.write("k_max: 50\n")
                f.write("k_div: 49\n")
                f.write("degree_max: 1\n")
                f.write("link: complete\n")
            f.close()
    # Generate running command
    for dataset in datasets:
        for kernel in kernels:
            csv_file = kernel + dataset + "_complete.csv"
            config_file = kernel + dataset + ".cfg"
            # run /Users/marco/Documents/Github/Thesis-Tools/a_t_i_o_colon.cfg
            print("sbt -mem 8048 \"run /Users/marco/Documents/Github/Thesis-Tools/" + config_file + "\"" + " > " + str(csv_file))
