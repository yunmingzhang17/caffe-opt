function [unchangedRatio, unchangedFeatureMapRatio, maxUnchangedRatioPerMap] = comparePoolIdxMap (prevPMIdX, prevPMIdY, currentPMIdX, currentPMIdY)

  sameX = abs(prevPMIdX - currentPMIdX);
  sameY = abs(prevPMIdY - currentPMIdY);
  sameXY = sameX + sameY;
  numSame = sum(sameXY(:) == 0);
  unchangedRatio = numSame/numel(prevPMIdX);

  
  [numRow, numCol, numDep, numImage] = size(prevPMIdX);

  unchangedFeatureMapCount = 0;
  unchangedRatioPerMap = zeros(numDep, numImage);

  %just for testing
  %numImage = 1; !!!!!!!!!
  for image = 1 : numImage
    for depth = 1 : numDep
      sameXY2D = sameXY(:,:,depth, image);
      if sum(sameXY2D(:)) == 0
	unchangedFeatureMapCount = unchangedFeatureMapCount + 1;
      end
      
      unchangedRatioPerMap(depth, image) = sum(sameXY2D(:) == 0)/numel(sameXY2D);
      
    end
  end

  unchangedFeatureMapRatio = unchangedFeatureMapCount/(numImage*numDep);
  maxUnchangedRatioPerMap = max(unchangedRatioPerMap(:));