import pandas as pd
import matplotlib.pyplot as plt
import argparse

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--filename', action="store", dest="filename", help='Filename ' )
parser.add_argument('--title', action="store", dest="title", help='title ', default='Title' )
parser.add_argument('--xlabel', action="store", dest="xlabel", help='xlabel ', default='xlabel' )
parser.add_argument('--scatter', action="store_true", dest="scatter", help='scatter ', default=False )
args = parser.parse_args()

dir = '/home/malcolm/uclan/output/csv/'
data = pd.read_csv(dir+args.filename, header=0).squeeze()

print(data)

y_label = data.columns[0]
y = data[y_label]
print(y)

markers = ['o', 'v', '+', '<', 'x', 'D', '*', 'X','o', 'v', '+', '<', 'x', 'D', '*', 'X']
for col in range(len(data.columns)-1):
    x_label = data.columns[col+1]
    x = data[x_label]
    print(x)
    if args.scatter:
        plt.scatter(x, y, label = x_label, marker=markers[col+1])
    else:
        plt.plot(x, y, label = x_label, marker=markers[col+1])

plt.title(args.title)
plt.xlabel(args.xlabel, fontsize=15)
plt.ylabel(y_label, fontsize=15)
if not args.scatter:
    plt.legend(loc='best')
plt.show()

