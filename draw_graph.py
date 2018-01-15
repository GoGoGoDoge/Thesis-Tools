import matplotlib.pylab as plt
import sys
import csv

if __name__ == "__main__":
    print("Start to draw figure")
    if len(sys.argv) < 2:
        print("Please Provide CSV file!")
        print("e.g. python3 draw_graph.py result.csv")

    filename = sys.argv[1]

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for row in reader:
            while len(data) < len(row):
                data.append([]);
            for idx, vals in enumerate(row):
                data[idx].append(float(vals))

        # Analysis the parsing data
        plt.plot(data[1], 'r-')
        plt.ylabel("NMI")
        plt.xlabel("nclusters")
        plt.show()
