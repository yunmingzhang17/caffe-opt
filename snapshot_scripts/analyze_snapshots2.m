clc
close all


addpath /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib;
snapshotDir ='/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num.txt'));

[s1, s2] = size(snapshotsNums);

numLayers = 8;
numElem = s2;
%stepSize = 4;
meanVector = zeros(numLayers, numElem);
iterVector = zeros(1, numElem);

c = caffeConfig(3);



for i = 1 : numElem
    snapshotFile = strcat(snapshotDir, 'caffenet_train_iter_', num2str(snapshotsNums(i)), '.caffemodel')
    c.definition_file = '/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/deploy.prototxt';
    c.binary_file  = snapshotFile;
    c = caffeInitialize(c);
    
    
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
    filename = strcat('layer', num2str(i));
    print(h, '-djpeg', filename);
end



