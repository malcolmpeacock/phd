# Locations of MIDAS weather stations.
#
# each has a 1 letter key a, b, c, etc.
#  then ... name:    identifies the station and used for the filename
#           coords:  latitude and longitude.
#           wind:    n=no, y=yes, s=sea (ie offshore)
#           pv:      n=no, y=yes
#           depth:   sea depth for wind=s only

def get_locations():

    locations = { "a" : { "name" : "dyfed_01198_aberporth", "coords" : [52.1391,-4.56999], "wind" : "y", "pv" : "y"  },
                  "b" : { "name" : "lancashire_01090_blackpool-squires-gate", "coords" : [53.7746,-3.03647], "wind" : "y", "pv" : "n" },
                  "c" : { "name" : "cornwall_01395_camborne", "coords" : [50.2178,-5.32656], "wind" : "y", "pv" : "y" },
                  "e" : { "name" : "kent_00744_east-malling", "coords" : [51.287,0.45006], "wind" : "n", "pv" : "y" },
                  "l" : { "name" : "moray-in-grampian-region_00137_lossiemouth", "coords" : [57.712,-3.322], "wind" : "y", "pv" : "n" },
                  "m" : { "name" : "marine_lat57.0_long1.8", "coords" : [57.0,1.8], "wind" : "s", "pv" : "n", "depth" : 50 },
                  "s" : { "name" : "lanarkshire_00982_salsburgh", "coords" : [55.8615,-3.87409], "wind" : "y", "pv" : "n" },
                  "u" : { "name" : "nottinghamshire_00554_sutton-bonington", "coords" : [52.8361,-1.24961], "wind" : "n", "pv" : "y" },
                  "w" : { "name" : "caithness_00032_wick-airport", "coords" : [58.4533,-3.0879], "wind" : "y", "pv" : "n" } }
    return locations
