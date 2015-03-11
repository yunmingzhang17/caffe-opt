function [unchangedRatio, unchangedFeatureMapRatio] = comparePoolIdxMap (prevPMIdX, prevPMIdY, currentPMIdX, currentPMIdY)

  sameX = abs(prevPMIdX - currentPMIdX);
  sameY = abs(prevPMIdY - currentPMIdY);
  sameXY = sameX + sameY;
  numSame = sum(sameXY(:) == 0);
  unchangedRatio = numSame/numel(prevPMIdX);

  unchangedFeatureMapRatio = 0;