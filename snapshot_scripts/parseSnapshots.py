import os
import sys
import subprocess
import re

#The script go through all the caffe model files and generate a file with all the iteration numbers

def analyzeDir(snapshotDir, output):

    outputFile = open(output, 'w')

    snapshotList = os.listdir(snapshotDir) 
    for snapshot in snapshotList:
        if snapshot.endswith('.caffemodel'):
            #print snapshot
            snapshotNum = re.findall(r'\d+', snapshot)[0]
            #print snapshotNum
            outputFile.write(snapshotNum + ' ')
    outputFile.close()

def analyzeDir2(snapshotDir, output):

    outputFile = open(output, 'w')

    snapshotList = os.listdir(snapshotDir) 
    for snapshot in snapshotList:
        if snapshot.endswith('.solverstate') and snapshot.startswith('caffe_object_train_iter_'):
            #print snapshot
            snapshotNum = re.findall(r'\d+', snapshot)[0]
            #print snapshotNum
            outputFile.write(snapshotNum + ' ')
    outputFile.close()


analyzeDir('/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/', 'output_snapshot_num.txt')

analyzeDir2('/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet', 'output_snapshot_num2.txt')
