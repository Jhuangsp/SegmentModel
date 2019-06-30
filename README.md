# SegmentModel
FY108 Skeleton-based action analysis - Action Segment Model

## Standard
In this implementation, we assume all keypoints share same moving period and no time-shift displacement.

Among this assumption, we define an **Action** to be a set of skeleton keypoints move together along a set fixed directions during a period of time.

And, how we label the Action's starting-ending points in a video is assigning the **Changing Point Frame** at all moving turning points, e.g. the frame that sitting on chair and stand straght in *StandSit video*.

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

<details>
<summary>First 2019-05-27</summary>

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
</details>


<details>
<summary>Add Sigmoid + Lower learning rate 2019-06-04</summary>

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
</details>


<details>
<summary>Add decay + Add weighted + Output 10 mid 2019-06-21</summary>

### Argument 2019-06-21
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 **(decay half at 60% & 80%)**
 - RNN size: 50
 - RNN layers: 4
 - **Input size: 20 frames**
 - **Output size: 10 frames**
 - **Weight: 20:1**

### Validation result
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-21/validall.png" alt="decay_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-21/validpart.png" alt="decay_part">
</p>
</details>


<details>
<summary>Add decay + Add weighted + Output 1 mid 2019-06-24</summary>

### Argument 2019-06-24
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 **(decay half at 60% & 80%)**
 - RNN size: 50
 - RNN layers: 4
 - **Input size: 21 frames**
 - **Output size: 1 frames**
 - **Weight: 20:1**

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-24/validall.png" alt="out1_decay_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-24/validpart.png" alt="out1_decay_part">
</p>
</details>


<details>
<summary>Add Batch normalize + Add Dropout 2019-06-25 (slow & bad)</summary>

### Argument 2019-06-25
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 **(decay half at 60% & 80%)**
 - RNN size: 50
 - RNN layers: 4
 - **Input size: 20 frames**
 - **Output size: 10 frames**
 - **Weight: 20:1**
 - **Batch normalize**
 - **Dropout: keep_rate = 0.9 (while training)**
Very slow (30M -> 2HR)
Loss dicrease faster at the early stage (2HR)

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25/all.png" alt="DO_BN_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25/part.png" alt="DO_BN_part">
</p>
</details>


<details>
<summary>Only add Dropout 2019-06-25_2</summary>

### Argument 2019-06-25_2
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 **(decay half at 60% & 80%)**
 - RNN size: 50
 - RNN layers: 4
 - **Input size: 20 frames**
 - **Output size: 10 frames**
 - **Weight: 20:1**
 - ~~Batch normalize~~
 - **Dropout: keep_rate = 0.9 (while training)**

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25_2/all.png" alt="DO_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25_2/part.png" alt="DO_part">
</p>
</details>


<details>
<summary>New Kernel 2019-06-30</summary>

### Argument 2019-06-30
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 (decay half at 60% & 80%)
 - RNN size: 50
 - RNN layers: 4
 - Input size: 20 frames
 - Output size: 10 frames
 - Weight: 20:1
 - Dropout: keep_rate = 0.9 (while training)
 - **Cancel the 3 different Gaussian Kernel**

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30/all.png" alt="new_kernal_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30/part.png" alt="new_kernal_part">
</p>
</details>


<details>
<summary>New Kernel + 30 Batch size 2019-06-30_2 (Bad)</summary>

### Argument 2019-06-30_2
 - Epochs: 50
 - Batch size: **30**
 - Learning rate: 0.0001 (decay half at 60% & 80%)
 - RNN size: 50
 - RNN layers: 4
 - Input size: 20 frames
 - Output size: 10 frames
 - Weight: 20:1
 - Dropout: keep_rate = 0.9 (while training)
 - **Cancel the 3 different Gaussian Kernel**

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_squat/all.png" alt="new_kernal_30_all">
</p>
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_2/part.png" alt="new_kernal_30_part">
</p>
</details>


<details>
<summary>New Kernel 2019-06-30_squat</summary>

### Argument 2019-06-30
 - Epochs: 50
 - Batch size: 15
 - Learning rate: 0.0001 (decay half at 60% & 80%)
 - RNN size: 50
 - RNN layers: 4
 - Input size: 20 frames
 - Output size: 10 frames
 - Weight: 20:1
 - Dropout: keep_rate = 0.9 (while training)
 - **Cancel the 3 different Gaussian Kernel**
 - **Change validation data to *Squat***

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_squat/all.png" alt="new_kernal_all">
</p>
</details>


<details>
<summary>New Kernel + 100 Epoch 2019-06-30_squat</summary>

### Argument 2019-06-30
 - Epochs: **100**
 - Batch size: 15
 - Learning rate: 0.0001 (decay half at 60% & 80%)
 - RNN size: 50
 - RNN layers: 4
 - Input size: 20 frames
 - Output size: 10 frames
 - Weight: 20:1
 - Dropout: keep_rate = 0.9 (while training)
 - **Cancel the 3 different Gaussian Kernel**
 - **Change validation data to *Squat***

### Validation result 
<p align="center">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_squat_100/all.png" alt="new_kernal_all">
</p>
</details>

## TODO
 - Technical documents (done)
 - 3 Method output
    - Probability: strengthen the non-zero field penalty (done)
    - Scale output: odd frames input, output 1 frame information (done)
    - Time error
 - ~~Try DTW Discrete Time Warping/Dynamic Time Warping and hidden markov model (by Rabiner)~~
 - Add decay (done)
 - Add dropout (done)
 - Add batch normalize (done)
 - ~~Data preprocess~~
 - Modify Gaussian Kernel (done)
 - Peak finding
 - Proper evaluation method
 - Auto hyperparameter finding
 - Bring back 3 diffrent Gaussian Kernel

