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

%layers = {'conv1', 'conv2', 'conv5', 'pool1',  'pool2',  'pool5'};
%idxOfLayerInNetwork = [2, 4, 9, 3, 6, 11]; %hard coded index into network set up!!!!!

layers = {'conv1', 'pool1'};
idxOfLayerInNetwork = [2, 3];

snapshotDir ='/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num2.txt'));
[s1, s2] = size(snapshotsNums);

numElem = 2; %just for testing

iterVector = zeros(1, numElem);


%snapshotPoolIdxMap = containers.Map();

snapshotPoolIdxMapX = containers.Map();
snapshotPoolIdxMapY = containers.Map();

deploy_txt ='/data/vision/torralba/datasetbias/caffe-latest/models/bvlc_reference_caffenet/deploy.prototxt'; 
binary_file ='/data/vision/torralba/datasetbias/caffe-latest/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'; 
verify_size = 0; 
[rf, rf_layers] = getReceptiveField(deploy_txt, binary_file, verify_size);

fid = fopen('analyze_pool_idx.txt', 'w');

for i = 1: numElem

    snapshotFile = strcat(snapshotDir, 'caffe_object_train_iter_', num2str(snapshotsNums(i)));
    c = caffeConfig(3);
    c.definition_file = '/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/object_deploy.prototxt';
    c.center_only = 1;
    c.reshape_features = 0;
    c.binary_file = snapshotFile;

    %dataset = tempname; dataset = dataset(5:15); %generating a tmp name 
    
    dataset = num2str(snapshotsNums(i));

    features= caffeFeatures(dataset, filelist, layers, c);

    %poolLayers = layers((numel(layers)/2 + 1), (numel(layers)));
    %convLayers = layers(1: (numel(layers)/2));



    poolMapX = containers.Map();
    poolMapY = containers.Map();

    %snapshotPoolIdxMap(snapshotFile) = poolMap; 
    
    snapshotPoolIdxMapX(snapshotFile) = poolMapX; 
    snapshotPoolIdxMapY(snapshotFile) = poolMapY; 


    for j = 1 : (numel(layers)/2)
    
      convLayer = features{j,1};
      poolIdx = (j+(numel(layers)/2));
      poolLayer = features{poolIdx, 1};
      [numRow, numCol, numDep, numImage] = size(poolLayer);  
      %[numRow, numCol, numDep, numImage] = size(convLayer)

      %hard coded for testing !!!!!!!
      numDep = 1;
      numImage = 1;

      convlayerName = char(layers(j));
      poolLayerName = char(layers(poolIdx));

      %poolIdxLayer = cell(size(poolLayer));
      
      poolIdxLayerX = zeros(size(poolLayer));
      poolIdxLayerY = zeros(size(poolLayer));


      for depth = 1 : numDep
	for imageIdx = 1 : numImage
	  for x = 1: numRow
	    for y = 1: numCol
	      
	      poolIdxInNetwork = idxOfLayerInNetwork(poolIdx);
	      convIdxInNetwork = idxOfLayerInNetwork(j);

	      maxVal = poolLayer(x,y, depth, imageIdx);
	      regionIdx = rf{poolIdxInNetwork}(x, y, 1:4);
	      x1 = regionIdx(1);
	      y1 = regionIdx(2);
	      x2 = regionIdx(3);
	      y2 = regionIdx(4);

	      region = convLayer(x1:x2, y1:y2, depth, imageIdx);
	      [rowIdx, colIdx] = findIdxOfMax(region, maxVal);

	      poolIdxLayerX(x,y, depth, imageIdx) = rowIdx;
	      poolIdxLayerY(x,y, depth, imageIdx) = colIdx;

	      %poolIdxLayer{x,y, depth, imageIdx} = [rowIdx, colIdx];
	      

	      
	      end
	    end
	  end
	end      

	%poolMap(poolLayerName) = poolIdxLayer;
	poolMapX(poolLayerName) = poolIdxLayerX;
	poolMapY(poolLayerName) = poolIdxLayerY;

	%size(find(poolIdxLayerX(:)))
	%size(find(poolIdxLayerY(:)))

    end
    
    
    
    if(i > 1)
      prevSnapshotFile = strcat(snapshotDir, 'caffe_object_train_iter_', num2str(snapshotsNums(i-1)));
      

      s1 = strcat('snapshot ', num2str(snapshotsNums(i)));
      fprintf(fid, '%s\n', s1); 
      prevPoolMapX = snapshotPoolIdxMapX(prevSnapshotFile);
      prevPoolMapY = snapshotPoolIdxMapY(prevSnapshotFile);
	      
      numPoolLayers = (numel(layers)/2);
     %numPoolLayers = 1; %hard coded for testing !!!!!!!!!

      for j = 1 : numPoolLayers
      	      poolIdx = (j+(numel(layers)/2));
      	      poolLayerName = char(layers(poolIdx));
	      fprintf(fid, '%s\n', poolLayerName);
      	      prevPMIdX = prevPoolMapX(poolLayerName);
      	      prevPMIdY = prevPoolMapY(poolLayerName);

              currentPMIdY = poolMapY(poolLayerName);
              currentPMIdX = poolMapX(poolLayerName);
	
	      fprintf(fid, 'pool map size is %d \n', size(prevPMIdx(:))); 
      	      
	      %numDiff = comparePoolIdxMap(prevPMIdx, currentPMIdx);
	      unchangedRatio = comparePoolIdxMap(prevPMIdX, prevPMIdY, currentPMIdX, currentPMIdY);	      
		      

              fprintf(fid, 'unchanged ratio is %s\n', unchangedRatio);
	      
      end
      
    end



end

fclose(fid);

%Testing for correctness for imagenet                                                    
poolMap = snapshotPoolIdxMap('/data/vision/torralba/datasetbias/caffe-latest/examples/imagenet/caffe_object_train_iter_10000');                                                                                                                          
%pm1 = poolMap('pool1');                                                                
% pm1{2,1,1,1}; %should be 2 1 (second row, first col)                                   
% pm1{1,1,1,1}; %should be 2,2                                                       
% pm5 = poolMap('pool5');                                                                
% pm5{4,2,1,1}; %should be 3,3                                                           
% pm5{5,2,1,1}; %should be 1,3                                                                                                                                                                                                                                      
%Analyze the change of pool 