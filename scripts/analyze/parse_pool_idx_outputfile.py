import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt

def parseResult(filename):
    
    
    file = open(filename)
    line = file.readline()

    snapshotVector = [];
    pool1 = []
    pool2 = []
    pool3 = []
    pools = [pool1, pool2, pool3]

    while line:
        print line
        words = line.split(' ')
        if len(words) == 1:
            ##line is snapshot number
            snapshot_num = re.findall(r'\d+', line)[0]
            #print snapshot_num
            snapshotVector.append(snapshot_num)
            for i in range(3):
                for j in range(5):
                    line = file.readline()
                #pool number
                words = line.split(' ')
                pools[i].append(float(words[3]))
            line = file.readline()
            
    print pool1
    plt.plot(snapshotVector, pool1, 'ro')
    plt.show()
    

filename = sys.argv[1]
parseResult(filename)
