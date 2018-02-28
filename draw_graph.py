import matplotlib.pylab as plt
import sys
import csv

if __name__ == "__main__":
    print("Start to draw figure")
    if len(sys.argv) < 2:
        print("Please Provide CSV file!")
        print("e.g. python3 draw_graph.py result.csv")

    filenames = []
    for idx, val in enumerate(sys.argv):
        if idx == 0:
            continue
        filenames.append(val)

    plt.tick_params(labelsize=15)

    for filename in filenames:
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            data = []
            for row in reader:
                # Filter out invalid data
                if len(row) < 9:
                    continue
                while len(data) < len(row):
                    data.append([]);
                for idx, vals in enumerate(row):
                    data[idx].append(float(vals))

            # Analysis the parsing data
            legend = filename.split(".csv")[0]
            # print(legend)
            plt.plot(data[2], data[3], label=legend)
            plt.ylabel("NMI", fontsize=18)
            plt.xlabel("k", fontsize=18)
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    plt.show()
