clc
close all


addpath /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib;
snapshotDir ='/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num.txt'));

[s1, s2] = size(snapshotsNums);

numLayers = 8;
numElem = s2;
sumAVector = zeros(numLayers, numElem);
iterVector = zeros(1, numElem);

c = caffeConfig(3);



for i = 1 : numElem
    snapshotFile = strcat(snapshotDir, 'caffenet_train_iter_', num2str(snapshotsNums(i)), '.caffemodel')
    c.definition_file = '/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/deploy.prototxt';
    c.binary_file  = snapshotFile;
    c = caffeInitialize(c);
    w = caffe('get_weights');
    for j = 1: numLayers
    %try it for the first few layer
        A = abs(w(j).weights{1});
        sumA = sum(sum(sum(sum(A))))
        sumAVector(j, i) = sumA;    
    end
    iterVector(i) = snapshotsNums(i);
end

for i = 1: numLayers
    h = figure;
    plot(iterVector, sumAVector(i,:)');
    filename = strcat('layer', num2str(i));
    print(h, '-djpeg', filename);
end



