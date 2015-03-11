addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/');
tmp1 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_sun397_pool5_giga.mat', 'imageList');
tmp2 = load('/data/vision/torralba/gigaSUN/deeplearning/dataset/feature_caffereference/DLfeature_ilsvrc2012test_pool5_giga.mat', 'imageList');
image_list = [tmp1.imageList; tmp2.imageList;];
clear tmp1 tmp2;
numTest = 500;
filelist = image_list(randperm(length(image_list), min(numTest, length(image_list))));
%layers = {'pool1', 'norm1', 'pool2', 'norm2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7'};
layers = {'pool5'}
snapshotFile ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/caffe_object_train_iter_450000';
c = caffeConfig(3);
c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
c.center_only = 1;
c.reshape_features = 1;
c.binary_file = snapshotFile;
dataset='imagenet-450000';
features= caffeFeatures(dataset, filelist, layers, c);
f = caffeLoad(dataset, layers, c);


c1 = caffeInitialize(c);
w = caffe('get_weights');

