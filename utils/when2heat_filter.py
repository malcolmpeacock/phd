# python script to read when2heat.csv from stdin and write only GB stuff to
# stdout
import sys

def getcols(line,country):
    cols = line.split(';')
    wanted_cols = []
    wanted_cols.append(0)
    nc=0
    for col in cols:
        nc+=1
        if col.startswith(country):
            wanted_cols.append(nc)
    return wanted_cols

def lineout(line,cols):
    parts = line.split(';')
    outline = ''
    for col in cols:
        outline += parts[col] + ';'
    return outline.replace(',','.')

year='2018'
country='GB' 
ln=0
cols=[]
for line in sys.stdin:
    if ln==0:
        ln+=1
        cols=newline=getcols(line,country)
        newline=lineout(line,cols)
        print(newline)
    else:
        if line[0:4]==year:
            newline=lineout(line,cols)
            print(newline)

