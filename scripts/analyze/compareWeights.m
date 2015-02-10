addpath(' /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib/');

snapshotDir ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/';



  c = caffeConfig(3);
  c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
  snapshotFile1 = strcat(snapshotDir, 'caffe_object_train_iter_', '100000') ;
  c.binary_file = snapshotFile1;
  c = caffeInitialize(c);
  w1 = caffe('get_weights');
  
  
 snapshotFile2 = strcat(snapshotDir, 'caffe_object_train_iter_', '400000') ;
  c = caffeConfig(3);
  c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';

  c.binary_file = snapshotFile2;
  c = caffeInitialize(c);
  w2 = caffe('get_weights');



  a = (w1(2).weights{1}(:,:,:,1) - w2(2).weights{1}(:,:,:,1));
  mean(abs(a(:)))
  sum(abs(a(:)))


snapshotDir = '/data/vision/scratch/torralba/khosla/cnn_dsl/imagenet_no_bp/'



  c = caffeConfig(3);
  c.definition_file = '/data/vision/scratch/torralba/khosla/cnn_dsl/imagenet_no_bp/deploy.prototxt';
  snapshotFile1 = strcat(snapshotDir, 'caffenet_train_iter_', '150000.caffemodel') ;
  c.binary_file = snapshotFile1;
  c = caffeInitialize(c);
  w3 = caffe('get_weights');
  
  
 snapshotFile2 = strcat(snapshotDir, 'caffenet_train_iter_', '400000.caffemodel') ;
  c = caffeConfig(3);

  c.definition_file = '/data/vision/scratch/torralba/khosla/cnn_dsl/imagenet_no_bp/deploy.prototxt';
  c.binary_file = snapshotFile2;
  c = caffeInitialize(c);
  w4 = caffe('get_weights');



  a = (w4(2).weights{1}(:,:,:,1) - w3(2).weights{1}(:,:,:,1));
  mean(abs(a(:)))
  sum(abs(a(:)))