import os
import sys
import subprocess
import re

#The script go through all the caffe model files and generate a file with all the iteration numbers

def analyzeDir(snapshotDir):

    outputFile = open('output_snapshot_num.txt', 'w')

    snapshotList = os.listdir(snapshotDir) 
    for snapshot in snapshotList:
        if snapshot.endswith('.caffemodel'):
            #print snapshot
            snapshotNum = re.findall(r'\d+', snapshot)[0]
            #print snapshotNum
            outputFile.write(snapshotNum + ' ')



analyzeDir('/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/')
