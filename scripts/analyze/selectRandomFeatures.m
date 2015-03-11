function [randomSelectedFeatures] = selectRandomFeatures (features)
  
  dim = size(features{1,1});
  numImages = dim(4);
  randId = randi(5);
  randomSelectedFeatures = cell(size(features));

  for layerId = 1: numel(features)
    featuresLayer = features{layerId,1};
    [x,y,depth, numImages] = size(featuresLayer);
    randomSelectFeaturesLayer = zeros(x,y,depth,numImages/10);
    for imageIdx = 1 : numImages
      if mod(imageIdx, 10) ~= randId
	continue;
      end
      
      outputImageIdx = ceil(imageIdx/10);
      randomSelectFeaturesLayer(:,:,:,outputImageIdx) = featuresLayer(:,:,:,imageIdx);
    end
    randomSelectedFeatures{layerId,1} = randomSelectFeaturesLayer;
  end
  
  clear(features);
end