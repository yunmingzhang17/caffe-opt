function [row, col] = findIdxOfMax(matrix, maxVal)
  matrix(matrix < 0) = 0; 
  [row,col] =  find(matrix == maxVal);
  if (isempty(row))
    disp('error in findIdxOfMax')
  end
end
