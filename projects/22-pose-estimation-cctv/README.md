# Pose-Estimation
pose estimation with Keypoint-RCNN and YOLOv8 on CCTV dataset

In this project, I intend to perform pose estimation using the keypoint-RCNN and YOLOv8 networks on a CCTV dataset. I will showcase the results of these two networks. The keypoint-RCNN network is a two-stage model for pose estimation and object detection, and it is based on keypoints. The architecture of this network is as follows:

![Keypoint-RCNN-neural-network-structure-mainly-consists-of-feature-extractor-region](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/ed05898c-1c33-44a9-9e13-d0143a25bc3f)

The YOLOv8 network is also a single-stage network and is used for both pose estimation and object detection. The structure of this network is as follows:


![1_YkkGwFBksWVbm4GmZfSDsg](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/69d26aa3-198e-4093-a99a-c5d07d0d556b)

The CCTV dataset consists of 700 images captured from surveillance cameras and is used for pose estimation and object detection on humans. I have divided this dataset into training, validation, and test sets, with 490 images for training, 105 images for validation, and 105 images for testing. The annotations for these three sets are provided in CSV files:

train_df: train annotation 

valid_df: validation annotation 

test_df: test annotation 

dataset link = <https://www.kaggle.com/datasets/jonathannield/cctv-human-pose-estimation-dataset>

Results of the Keypoint-RCNN network after training:

![download (27)](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/dcc84624-316d-4de0-b2b5-9f949a5bdefb)

![download (26)](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/ada37cc7-6223-470a-b235-4142e63712f3)

![download (25)](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/dcfce5a4-34f4-4ee9-a91b-845118947db7)

Results of the YOLO-v8 network after training:

![download (28)](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/228664dc-00cd-4595-9f97-66fa742914d3)

![download (29)](https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/bf45630e-56ac-46b5-8ee5-578b33dc7d3a)

YOLO-v8 on video:

https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/d0137e0e-a9ec-4926-91a4-1b3787248248




https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/f28166cd-3932-4b8b-8e94-af5fce78059e




https://github.com/ahmadrezabaqerzade/Pose-Estimation/assets/94822419/e29d6643-3953-47ee-8008-3cd4eb2af48f








