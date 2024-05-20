
# Visualize the results

## Updates

Debug attempts

- [x] Check the `POINT_CLOUD_RANGE` config.
- [x] Check the `transform_points_to_voxels` config.
- [x] Check the `ANCHOR_GENERATOR_CONFIG` config.
- [ ] Check the `EVAL_METRIC`.
- [x] Check if the `gt` data contains the labels out of lidar vision(out of range), [source](https://github.com/open-mmlab/OpenPCDet/issues/805).
  - Yes, we do have out-of-range points in our `infos` files. I cannot ensure it is unnormal.
- [ ] Check if the `gt` data and `infos` files has expected format.

2024.5.20

We updated the config files, they are looking better now. Now, we got slightly higher acc on all tasks. But the bev and 3d acc is still poor.

```log
Car AP@0.70, 0.70, 0.70:
bbox AP:83.3104, 83.3104, 83.3104
bev  AP:9.0233, 9.0233, 9.0233
3d   AP:4.5455, 4.5455, 4.5455
aos  AP:59.84, 59.84, 59.84
```

We think it could be related to the kitti `EVAL_METRIC` that we are using in our model config.

## Python

- gbin.py: get the dataset pc range config
- grange.py: get the model pc range config
- ganc.py: get the anchor configs
- show_gtd.py: get the first sample of gt-database

## TensorBoard

Check and select the env.

```bash
conda info --env
conda activate pytorch-mmdet3d
set DISPLAY=:0 
```

Install tensorboard

```bash
conda activate base
pip install tensorboard
```

Use tensorboard

train

```bash
tensorboard --logdir=output/tensorboard
```

eval

```bash
tensorboard --logdir=output/eval/eval_with_train/tensorboard_val
```

## Log file - Explanation of Evaluation Metrics

### Training Phase

During the training phase, the following information is logged:

```log
2024-05-16 20:12:52,737   INFO  Train:   80/80 (100%) [ 859/960 ( 89%)]  Loss: 0.6325 (0.825)  LR: 6.556e-08  Time cost: 00:36/00:04 [52:18/00:04]  Acc_iter 76700       Data time: 0.00(0.00)  Forward time: 0.04(0.04)  Batch time: 0.04(0.04)
2024-05-16 20:12:54,707   INFO  Train:   80/80 (100%) [ 909/960 ( 95%)]  Loss: 0.4804 (0.827)  LR: 3.907e-08  Time cost: 00:38/00:02 [52:20/00:02]  Acc_iter 76750       Data time: 0.00(0.00)  Forward time: 0.04(0.04)  Batch time: 0.04(0.04)
2024-05-16 20:12:56,624   INFO  Train:   80/80 (100%) [ 959/960 (100%)]  Loss: 0.9327 (0.826)  LR: 3.000e-08  Time cost: 00:40/00:00 [52:22/00:00]  Acc_iter 76800       Data time: 0.00(0.00)  Forward time: 0.03(0.04)  Batch time: 0.04(0.04)
```

- **Epoch**: 80/80 (100%) indicates the completion of the 80th epoch out of 80.
- **Iteration**: [859/960 (89%)] indicates the 859th iteration out of 960 in the current epoch (89% completion).
- **Loss**: Current loss is 0.6325, with an average loss of 0.825 over the epoch.
- **Learning Rate** (LR): The learning rate is 6.556e-08.
- **Time Cost**: Time taken for the current batch is 00:36, with 00:04 remaining. Total epoch time is 52:18, with 00:04 remaining.
- **Accumulated Iterations** (Acc_iter): 76700 iterations have been completed.
- **Data Time, Forward Time, Batch Time**: Average times for data loading, forward pass, and batch processing are 0.00, 0.04, and 0.04 seconds, respectively.

### Evaluation Phase

```log
2024-05-16 20:12:56,986   INFO  **********************Start evaluation user/sun.qumeng/u11423/OpenPCDet/tools/cfgs/custom_models/pps(default)**********************
2024-05-16 20:12:57,183   INFO  Loading Custom dataset.
2024-05-16 20:12:57,233   INFO  Total samples for CUSTOM dataset: 480
2024-05-16 20:12:57,241   INFO  ==> Loading parameters from checkpoint /home/uni08/hpc/sun.qumeng/u11423/OpenPCDet/output/user/sun.qumeng/u11423/OpenPCDet/tools/cfgs/custom_models/pps/default/ckpt/checkpoint_epoch_80.pth to GPU
2024-05-16 20:12:57,415   INFO  ==> Checkpoint trained from version: pcdet+0.6.0+37a88e1+py10cc925
2024-05-16 20:12:57,419   INFO  ==> Done (loaded 127/127)
```

- Dataset Loading: Indicates the loading of the custom dataset with a total of 480 samples.
- Checkpoint Loading: Parameters are loaded from the checkpoint file checkpoint_epoch_80.pth to the GPU.
- Checkpoint Version: The checkpoint was trained from version pcdet+0.6.0+37a88e1+py10cc925.

#### Evaluation Results

```log
2024-05-16 20:13:00,121   INFO  *************** Performance of EPOCH 80 *****************
2024-05-16 20:13:00,122   INFO  Generate label finished(sec_per_example: 0.0056 second).
2024-05-16 20:13:00,122   INFO  recall_roi_0.3: 0.000000
2024-05-16 20:13:00,122   INFO  recall_rcnn_0.3: 0.102792
2024-05-16 20:13:00,122   INFO  recall_roi_0.5: 0.000000
2024-05-16 20:13:00,122   INFO  recall_rcnn_0.5: 0.032780
2024-05-16 20:13:00,122   INFO  recall_roi_0.7: 0.000000
2024-05-16 20:13:00,122   INFO  recall_rcnn_0.7: 0.006070
2024-05-16 20:13:00,122   INFO  Average predicted number of objects(480 samples): 10.829
```

- Label Generation: Finished with an average time of 0.0056 seconds per example.
- Recall Metrics:
  - recall_roi_0.3: 0.000000
  - recall_rcnn_0.3: 0.102792
  - recall_roi_0.5: 0.000000
  - recall_rcnn_0.5: 0.032780
  - recall_roi_0.7: 0.000000
  - recall_rcnn_0.7: 0.006070
- Average Predicted Number of Objects: 10.829 per sample (480 samples).

#### Car, Pedestrian, and Truck Metrics

For each category (such as Car, Pedestrian, Truck), several key metrics are evaluated under different criteria:

- bbox AP (2D bounding box Average Precision): The average precision of 2D bounding boxes in the image plane.
- bev AP (Bird's Eye View Average Precision): The average precision from a bird's eye view perspective.
- 3d AP (3D bounding box Average Precision): The average precision of 3D bounding boxes in the 3D space.
- aos AP (Average Orientation Similarity): The average precision of orientation similarity.

These metrics are evaluated at different IoU (Intersection over Union) thresholds.

##### CAR

- AP@0.70, 0.70, 0.70: Evaluated at IoU threshold of 0.70.
  - bbox AP: 78.9882 indicates an average precision of 78.9882 for 2D bounding boxes.
  - bev AP: 0.1783 indicates an average precision of 0.1783 from a bird's eye view.
  - 3d AP: 0.1783 indicates an average precision of 0.1783 for 3D bounding boxes.
  - aos AP: 59.48 indicates an average orientation similarity precision of 59.48.
- AP_R40@0.70, 0.70, 0.70: Evaluated using the R40 method (40 sampling points) at IoU threshold of 0.70.
  - bbox AP: 83.6878 indicates an average precision of 83.6878 for 2D bounding boxes.
  - bev AP: 0.0327 indicates an average precision of 0.0327 from a bird's eye view.
  - 3d AP: 0.0200 indicates an average precision of 0.0200 for 3D bounding boxes.
  - aos AP: 60.39 indicates an average orientation similarity precision of 60.39.
- Similarly, AP@0.70, 0.50, 0.50 represents evaluation results at an IoU threshold of 0.50.

##### PEDESTRIAN

- Pedestrian evaluated at an IoU threshold of 0.50.
  - bbox AP: 14.5933 indicates an average precision of 14.5933 for 2D bounding boxes.
  - bev and 3d AP: 0.0000 indicate no correct detections in bird's eye view and 3D space.

##### TRUCK

- Truck evaluated at an IoU threshold of 0.70.
  - bbox AP: 27.2727 indicates an average precision of 27.2727 for 2D bounding boxes.
  - bev and 3d AP: 0.0000 indicate no correct detections in bird's eye view and 3D space.

#### Summary

These evaluation metrics help you understand the performance of your model for different categories and evaluation criteria. For example, the model performs well for 2D bounding box detection in the Car category but poorly for 3D space detection. Similarly, the 3D detection performance for Pedestrian and Truck categories is also poor. These metrics provide insights for further analysis and improvement of the model.