import sys

if __name__ == '__main__':
  clustering = input("Please enter clustering methods (hac, kmeans, kmedoids, spec):")
  kernel_name = ['ati','atr','cpi','cpr','spi','spr','ccti','cctr','sfi']
  kernel_file = ['a_t_i', 'a_t_r', 'c_p_i', 'c_p_r', 's_p_i', 's_p_r', 'c_ct_i', 'c_ct_r', 's_f_i']
  dataset_name = input("Please enter dataset name(web, moschitti, leukemia, cystic, colon):")
  for degree in range(2, 11):
    filename = str(dataset_name) + "_" + str(clustering) + "_" + str(degree) + ".sh"
    print("editing " + filename + " ...")
    with open(filename, 'w') as f:
      for i, value in enumerate(kernel_name):
        command = "python3 kernel_grid_search_" + str(clustering) + "_nmi_tune.py " + str(kernel_file[i]) + "_o_" + str(dataset_name) + ".kernel" + " yes 0.1 " + str(degree) + " > " + str(dataset_name) + "_" + str(kernel_name[i]) + "_" + str(clustering) + "_beta_tuned_d" + str(degree) + ".csv\n"
        f.write(command)
  
