addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/');
addpath('/data/vision/scratch/torralba/khosla/cnn_dsl');
addpath('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/scripts/analyze');

%getting the test imageset
tmp1 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_sun397_pool5_giga.mat', 'imageList');
tmp2 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_ilsvrc2012test_pool5_giga.mat', 'imageList');

image_list = [tmp1.imageList; tmp2.imageList;];
clear tmp1 tmp2;
numTest = 500; %a minimum of 500
filelist = image_list(randperm(length(image_list), min(numTest, length(image_list))));

layers = {'conv1', 'conv2', 'conv5', 'pool1',  'pool2',  'pool5'};


snapshotDir ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num2.txt'));
[s1, s2] = size(snapshotsNums);
numElem = 1;
iterVector = zeros(1, numElem);


for i = 1: numElem

    snapshotFile = strcat(snapshotDir, 'caffe_object_train_iter_', num2str(snapshotsNums(i)))
    c = caffeConfig(3);
    c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
    c.center_only = 1;
    c.reshape_features = 0;
    c.binary_file = snapshotFile;

    %dataset = tempname; dataset = dataset(5:15); %generating a tmp name 
    
    dataset = 'test2';

    features= caffeFeatures(dataset, filelist, layers, c);

    %poolLayers = layers((numel(layers)/2 + 1), (numel(layers)));
    %convLayers = layers(1: (numel(layers)/2));

    for j = 1 : (numel(layers)/2)
    
      convLayer = features{j,1};
      poolIdx = (j+(numel(layers)/2));
      poolLayer = features{poolIdx, 1};
      %[numRow, numCol, numDep, numImage] = size(poolLayer)  
      %[numRow, numCol, numDep, numImage] = size(convLayer)
      convlayerName = layers(j)
      poolLayerName = layers(poolIdx)
      
    end
    
    
end
