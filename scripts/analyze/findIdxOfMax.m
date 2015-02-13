function [row, col] = findIdxOfMax(matrix, maxVal)
  matrix(matrix < 0) = 0; 
  [row,col] =  find(matrix == maxVal,1, 'first');
  if (isempty(row))
    disp('error in findIdxOfMax')
  end
end
