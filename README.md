# SegmentModel
FY108 Skeleton-based action analysis - Action Segment Model

## Structure
<p align="center">
  <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/SegmentStructure.png" alt="SegmentStructure">
</p>

## Data

### Input Data
Skeleton detected by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) COCO model.
#### Normalize
Normaliaze the joint coordinate which take *Neck(idx:1)* as **origin**, **distance** from *Neck(idx:1)* to *Nose(idx:0)* as length unit.
<p align="center">
  <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/InputData.png" alt="InputData">
</p>

### Output Data
We labeled all the **Changing Point Frame** to 1, others to 0. The **Changing Point Frame** means those frames which can separate one action into two sub-actions.
<p align="center">
  <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/GTData.png" alt="GTData">
</p>
