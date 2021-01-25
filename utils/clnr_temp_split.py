# python script to investigate cnlr heat pump data.

clnr_filename = "TemperatureData.csv"
clnr_dir = "/home/malcolm/uclan/data/CLNR-L082-TC12-dataset-March-15/TC3"
#clnr_filename = "SmallMonitorData.csv"
#clnr_filename = "trial.15017.csv"
filename = clnr_dir + '/' + clnr_filename

locations={}

# Opening file 
fin = open(filename, 'r') 
count = 0
for line in fin: 
    count += 1
    # skip header
    if count>1:
        values = line.split(',')
        location = values[0]
        name = values[1]
        if name == 'External temperature':
            varname = 'tempout'
        else:
            if name =='Zone 1 temperature':
                varname = 'tempin'
            else:
                print("Unknown variable:{}".format(name) )
                varname = 'none'
        time = values[3]
        value = values[4]
        if count%500==0:
            print("Count {} Location {} Var {}".format(count, location, varname) )
        filename = "{}/files/{}_{}.csv".format(clnr_dir,location,varname)
        if location in locations:
            if varname in locations[location]:
                fout = locations[location][varname]
            else:
                fout = open(filename, "w")
                fout.write("time, value\n")
                locations[location][varname] = fout
        else:
            fout = open(filename, "w")
            fout.write("time, value\n")
            locations[location] = {varname: fout}
        fout.write("{}, {}".format(time, value))
  
# Closing files 
for location in locations:
    varnames = locations[location]
    for varname in varnames:
        fout = varnames[varname]
        fout.close() 
