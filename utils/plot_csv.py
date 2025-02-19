import pandas as pd
import matplotlib.pyplot as plt
import argparse

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--filename', action="store", dest="filename", help='Filename ' )
parser.add_argument('--title', action="store", dest="title", help='title ', default='Title' )
parser.add_argument('--xlabel', action="store", dest="xlabel", help='xlabel ', default='xlabel' )
parser.add_argument('--scatter', action="store_true", dest="scatter", help='scatter ', default=False )
parser.add_argument('--ycol', action="store", dest="ycol", help='ycol ', default=None )
parser.add_argument('--yunits', action="store", dest="yunits", help='yunits ', default=None )
parser.add_argument('--xcols', action="store", dest="xcols", help='xcols ', default=None )
parser.add_argument('--legends', action="store", dest="legends", help='legends ', default=None )
parser.add_argument('--exclude', action="store", dest="exclude", help='exclude ', default=None )
args = parser.parse_args()

dir = '/home/malcolm/uclan/output/csv/'
markers = ['o', 'v', '+', '<', 'x', 'D', '*', 'X','o', 'v', '+', '<', 'x', 'D', '*', 'X']
styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed', 'dashdot', 'dashdot', 'solid', 'dotted', 'dashed' ]
colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'yellow', 'salmon' ]

data = pd.read_csv(dir+args.filename, header=0).squeeze()

# print(data)

excludes=[]
if args.exclude:
    excludes = args.exclude.split(',')

if not args.ycol:
    y_label = data.columns[0]
    print('First col  default')
else:
    y_label = args.ycol
    print('col from arg ')
print('Y column: {}'.format(y_label) )

if not args.xcols:
    print('All x cols  default')
    xcols = []
    for col in range(len(data.columns)-1):
        xcols.append(data.columns[col+1])
else:
    xcols = args.xcols.split(',')
    print('x cols passed in')


# Extract scenarios
x=[]
if 'scenario' in data.columns:
    # Add Extra variables
    data['lostp'] = data['lost'] / data['energy']
    data['slostp'] = data['slost'] / data['energy']
    data['lostd'] = data['lost'].diff()
    data['slostd'] = data['slost'].diff()
    scount=0
    print('Grouping by scenario')
    group = data.groupby('scenario')
    for scenario, contents in group:
        if not scenario in excludes:
#           print(contents)
            y = contents[y_label]
            col=0
            ncols = len(xcols)
            for x_label in xcols:
                x = contents[x_label]
                if ncols == 1:
                    label = '{}'.format(scenario)
                else:
                    if args.legends:
                        legends = args.legends.split(',')
                        label = '{} : {}'.format(scenario,legends[col])
                    else:
                        label = '{} : {}'.format(scenario,x_label)
                print('Label : {} '.format(label) )
                if args.scatter:
                    plt.scatter(x, y, label = label, marker=markers[col])
                else:
                    plt.plot(x, y, label = label, marker=markers[col], color=colours[scount])
                col+=1
            scount+=1

else:
    y = data[y_label]
    col=0
    for x_label in xcols:
        col+=1
        x = data[x_label]
        print('X column: {}'.format(x_label) )
        if args.scatter:
            plt.scatter(x, y, label = x_label, marker=markers[col])
        else:
            plt.plot(x, y, label = x_label, marker=markers[col])

plt.title(args.title)
plt.xlabel(args.xlabel, fontsize=15)
if args.yunits:
    y_label += args.yunits
plt.ylabel(y_label, fontsize=15)
if not args.scatter:
    plt.legend(loc='best')
plt.show()

