clc
close all


addpath /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib;
%snapshotDir ='/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/';
%snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num.txt'));

snapshotDir ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num2.txt'));

[s1, s2] = size(snapshotsNums);

numLayers = 8;
numElem = s2;
%stepSize = 4;
meanVector = zeros(numLayers, numElem);
sumAVector = zeros(numLayers, numElem);
iterVector = zeros(1, numElem);
iterVectorSum = zeros(1, numElem);
c = caffeConfig(3);



for i = 1 : numElem
    snapshotFile = strcat(snapshotDir, 'caffe_object_train_iter_', num2str(snapshotsNums(i)));
    %c.definition_file = '/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/deploy.prototxt';
    c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
    c.binary_file  = snapshotFile;
    c = caffeInitialize(c);
    w = caffe('get_weights');
    for j = 1: numLayers
        A = abs(w(j).weights{1});
        sumA = sum(A(:));
        sumAVector(j, i) = sumA;    
    end
    iterVectorSum(i) = snapshotsNums(i);
    
    if (i == 1)
        lastw= caffe('get_weights');
        continue;
    else
        currentw = caffe('get_weights');
        for j = 1: numLayers
        %try it for the first few layer
        % the number is not in percentage, it is absolute change ratio
            diff = abs(currentw(j).weights{1} - lastw(j).weights{1});
            meanVal = mean(abs(currentw(j).weights{1}(:)));
            diff = diff/meanVal; 
            meanDiff = mean(diff(:));
            %meanDiff = mean(mean(mean(mean(diff))));
            meanVector(j, i) = meanDiff*100;            
                       
        end
        lastw = currentw;
    end
    iterVector(i) = snapshotsNums(i);
end

for i = 1: numLayers
    h = figure;
    plot(iterVector, meanVector(i,:)');
    ylim([0,100]);
    filename = strcat('weight-change-percent-layer', num2str(i));
    title(filename);
    xlabel('iterations');
    ylabel('change in percentage');
    print(h, '-djpeg', filename);
end

for i = 1: numLayers
    h = figure;
    plot(iterVectorSum, sumAVector(i,:)');
    filename = strcat('weight-abs-layer', num2str(i));
    title(filename);
    xlabel('iterations');
    ylabel('absolute sum of weight');
    print(h, '-djpeg', filename);
end



