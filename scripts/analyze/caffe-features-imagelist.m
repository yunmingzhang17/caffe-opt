addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/'); 
tmp1 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_sun397_pool5_giga.mat', 'imageList');
tmp2 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_ilsvrc2012test_pool5_giga.mat', 'imageList');
image_list = [tmp1.imageList; tmp2.imageList;];
clear tmp1 tmp2;
numTest = 2; 
filelist = image_list(randperm(length(image_list), min(numTest, length(image_list))));
c = caffeConfig(3);
c.center_only = 1;
layers = {'fc6', 'fc7'};
dataset = 'test';
features= caffeFeatures(dataset, filelist, layers, c);
f = caffeLoad(dataset_name, layers, c);  %cell array, each being a cell of 2x4096
f1 = f{1,1};
f2 = f{2,1};

length(find(f1 == 0)) %the number of zeros in f1  6726
length(find(f2 == 0)) %the number of zeros in f2  6535