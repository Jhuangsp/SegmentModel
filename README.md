# SegmentModel
FY108 Skeleton-based action analysis - Action Segment Model

## Standard
In this implementation, we assume all keypoints share same moving period and no time-shift displacement.

Among this assumption, we define an **Action** to be a set of skeleton keypoints move together along a set fixed directions during a period of time.

And, how we label the Action's starting-ending points in a video is assigning the **Changing Point Frame** at all moving turning points, e.g. the frame that sitting on chair and stand straght in *tandSit video*.

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

## Result

### Argument 2019-05-27
 - Epochs: 100
 - Batch size: 15
 - Learning rate: 0.001
 - RNN size: 50
 - RNN layers: 4
 - Frames of 1 Input Sequence: 20

### Validation result (Bad performance)
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/valid_result.png" alt="valid_result">
</p>

### Training result
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/train_result.png" alt="train_result">
</p>


### Argument 2019-06-04
 - Epochs: 100
 - Batch size: 15
 - Learning rate: **0.0001**
 - RNN size: 50
 - RNN layers: 4
 - Frames of 1 Input Sequence: 20
 - **Add sigmoid**

### Validation result (Bad performance)
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/after_sigmoid_lr1e-4_valid.png" alt="after_sigmoid_lr1e-4_valid">
</p>

### Training result
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/after_sigmoid_lr1e-4_train.png" alt="after_sigmoid_lr1e-4_train">
</p>

## TODO
 - Modify to read 1 frame output 1 result
     - read 1 frame and output 1 frame's possibility of being Changing Point Frame
     - do not reset the model state after output 1 frame's result
     - When to update parameters while training? after whole video (need more data)? is it good?
     - Is it good that output 1 scale possibility value? or output boolean?
 - Try DTW Discrete Time Warping/Dynamic Time Warping and hidden markov model (by Rabiner)
 - LR decay
 - Add dropout
 - Add batch normalize
 - Auto hyperparameter finding

