addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/'); 
tmp1 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_sun397_pool5_giga.mat', 'imageList');
tmp2 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_ilsvrc2012test_pool5_giga.mat', 'imageList');
image_list = [tmp1.imageList; tmp2.imageList;];
clear tmp1 tmp2;
numTest = 1000; 
filelist = image_list(randperm(length(image_list), min(numTest, length(image_list))));
c = caffeConfig(3);
c.center_only = 1;
c.reshape_features = 1;
%layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
layers = {'pool1', 'norm1', 'pool2', 'norm2', 'conv3'};
dataset = tempname; dataset = dataset(5:15);
features= caffeFeatures(dataset, filelist, layers, c);
f = caffeLoad(dataset, layers, c);  %cell array, each being a cell of 2x4096

for i=1:size(f,1)
	f1 = f{i, 1};
disp(['layer: ' layers{i} ', sparsity: ' num2str(mean(f1(:)==0))]);
end


