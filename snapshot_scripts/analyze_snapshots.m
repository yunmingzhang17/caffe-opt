
addpath /data/vision/torralba/regionmem/memorability_cnn/lib/caffeLib;
snapshotDir ='/data/vision/scratch/torralba/khosla/deep_train/imagenet_iter/*.caffemodels';
snapshotsNums = sort(importdata('/data/vision/scratch/torralba/khosla/cnn_dsl/caffe/snapshot_scripts/output_snapshot_num.txt'));

[s1, s2] = size(snapshotsNums);
for i = 1 : s2
    snapshotName = strcat('caffenet_train_iter_', num2str(i), '.caffemodel')
end
    