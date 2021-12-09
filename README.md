# SGNet
3D object detection combining semantic and geometric features from point clouds

## environment:

    Linux (tested on Ubuntu 20.04 )
    Python 3.7
    PyTorch 1.9
    CUDA 11.1

## Results on KITTI test set

[Submission link](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=7ed280afe29100628a32d961d66f3133bfe0d077)

## Results on KITTI val set
```
Car AP@0.70, 0.70, 0.70:
bbox AP:97.6366, 89.4210, 89.1686
bev  AP:90.2512, 87.9546, 87.7203
3d   AP:89.0807, 85.1031, 78.7955
aos  AP:97.61, 89.33, 89.01
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.7089, 94.9175, 92.7567
bev  AP:95.5359, 91.2230, 89.2623
3d   AP:92.1615, 85.0114, 82.9931
aos  AP:98.69, 94.78, 92.56
 
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:76.8437, 71.0767, 64.3539
bev  AP:69.4794, 61.9789, 58.1103
3d   AP:68.2862, 60.6319, 56.7598
aos  AP:75.56, 69.54, 62.79
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:77.4240, 71.7943, 65.7034
bev  AP:70.8647, 62.8840, 57.5485
3d   AP:68.6644, 60.3866, 55.0046
aos  AP:76.01, 70.07, 63.85
 
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:95.2546, 83.0810, 81.9383
bev  AP:93.2908, 78.5258, 72.8116
3d   AP:90.8925, 72.2304, 69.8454
aos  AP:95.11, 82.80, 81.65
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:96.7342, 84.5629, 81.9806
bev  AP:95.0077, 78.7425, 75.5409
3d   AP:92.1955, 74.3969, 70.1763
aos  AP:96.61, 84.29, 81.70
 
```

# Acknowledgement
Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Thanks OpenMMLab Development Team for their awesome codebases.
