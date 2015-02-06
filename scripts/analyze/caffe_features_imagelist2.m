addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/'); 
tmp1 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_sun397_pool5_giga.mat', 'imageList');
tmp2 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_ilsvrc2012test_pool5_giga.mat', 'imageList');
image_list = [tmp1.imageList; tmp2.imageList;];
clear tmp1 tmp2;
numTest = 2; 
filelist = image_list(randperm(length(image_list), min(numTest, length(image_list))));

%layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
layers = {'pool1', 'norm1', 'pool2', 'norm2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'};


snapshotDir ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num2.txt'));
[s1, s2] = size(snapshotsNums);
numElem = s2;
iterVector = zeros(1, numElem);


for i = 1 : numElem
  snapshotFile = strcat(snapshotDir, 'caffe_object_train_iter_', num2str(snapshotsNums(i)))
  c = caffeConfig(3);
  c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
  c.center_only = 1;
  c.reshape_features = 1;
  c.binary_file = snapshotFile;

  dataset = tempname; dataset = dataset(5:15); %generating a tmp name
  features= caffeFeatures(dataset, filelist, layers, c);
  f = caffeLoad(dataset, layers, c);  

  disp(['snapshot '  num2str(snapshotsNums(i))]);
  for i=1:size(f,1)
	f1 = f{i, 1};
  disp(['layer: ' layers{i} ', sparsity: ' num2str(mean(f1(:)==0))]);


  end
end

