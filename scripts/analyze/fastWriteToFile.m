function [] = fastWriteToFile(filename, matrix)

  matrix = matrix';
  fid = fopen(filename, 'w+');
  for i=1:size(matrix, 1)
    fprintf(fid, '%f ', matrix(i,:));
    fprintf(fid, '\n');
end
  fclose(fid);

end