# SegmentModel
FY108 Skeleton-based action analysis - Action Segment Model

## Structure
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/SegmentStructure.png" alt="SegmentStructure">
</p>

## Data

### Input Data
Skeleton detected by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) COCO model.
#### Normalize
Normaliaze the joint coordinate which take *Neck(idx:1)* as **origin**, **distance** from *Neck(idx:1)* to *Nose(idx:0)* as length unit.
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/InputData.png" alt="InputData">
</p>

### Output Data
We labeled all the **Changing Point Frame** to 1, others to 0. The **Changing Point Frame** means those frames which can separate one action into two sub-actions.
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GTData.png" alt="GTData">
</p>

#### Gaussian Weighting
We do not directly use the target as ground truth, we weight the target with [Discrete Gaussian Kernel](https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel) for each times of Fine Tune. (In our task we Fine Tune 3 times, so we have 3 different size Discrete Gaussian Kernel)

##### Original Target
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/Target.png" alt="Target" width="700">
</p>

##### 3 different size Discrete Gaussian Kernel
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GaussianWeight.png" alt="GaussianWeight" width="400">
</p>

##### 3 Weighted Target for 3 times Fine Tune
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GaussianWeightedTarget.png" alt="GaussianWeightedTarget" width="700">
</p>