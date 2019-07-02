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
<p align="center" width="200">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/InputData.png" alt="InputData">
</p>

### Output Data
We labeled all the **Changing Point Frame** to 1, others to 0. The **Changing Point Frame** means those frames which can separate one action into two sub-actions.
<p align="center" width="200">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GTData.png" alt="GTData">
</p>

#### Gaussian Weighting
We do not directly use the target as ground truth, we weight the target with [Discrete Gaussian Kernel](https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel) for each times of Fine Tune. (In our task we Fine Tune 3 times, so we have 3 different size Discrete Gaussian Kernel)

##### Original Target
<p align="center" width="200">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/Target.png" alt="Target" width="700">
</p>

##### 3 different size Discrete Gaussian Kernel
<p align="center" width="150">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GaussianWeight.png" alt="GaussianWeight" width="400">
</p>

##### 3 Weighted Target for 3 times Fine Tune
<p align="center" width="200">
    <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/GaussianWeightedTarget.png" alt="GaussianWeightedTarget" width="700">
</p>

## Result

<details>
    <summary>(Old) First 2019-05-27</summary>
    Argument 2019-05-27
        
        - Epochs: 100
        - Batch size: 15
        - Learning rate: 0.001
        - RNN size: 50
        - RNN layers: 4
        - Frames of 1 Input Sequence: 20
Validation result (Bad performance)
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/valid_result.png" alt="valid_result">
    </p>
Training result
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/train_result.png" alt="train_result">
    </p>
</details>

<details>
    <summary>(Old) Add Sigmoid + Lower learning rate 2019-06-04</summary>
    Argument 2019-06-04
        
        - Epochs: 100
        - Batch size: 15
        - Learning rate: **0.0001**
        - RNN size: 50
        - RNN layers: 4
        - Frames of 1 Input Sequence: 20
        - **Add sigmoid**
Validation result (Bad performance)
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/after_sigmoid_lr1e-4_valid.png" alt="after_sigmoid_lr1e-4_valid">
    </p>
Training result
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/after_sigmoid_lr1e-4_train.png" alt="after_sigmoid_lr1e-4_train">
    </p>
</details>

---------------------------------------------------------------------------

Add learning rate **decay**, and strengthen the change point contribution by adding a higher **weight** to the non-zero field.
<details>
    <summary>Add decay + Add weighted + Output 10 2019-06-21</summary>
    Argument 2019-06-21

         - Epochs: 50
         - Batch size: 15
         - Learning rate: 0.0001 **(decay half at 60% & 80%)**
         - RNN size: 50
         - RNN layers: 4
         - **Input size: 20 frames**
         - **Output size: 10 frames**
         - **Weight: 20:1**
Validation result
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-21/validall.png" alt="decay_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-21/validpart.png" alt="decay_part">
    </p>
</details>

Try the second output type, witch only output only **one middle** frame.
<details>
    <summary>Add decay + Add weighted + Output 1 2019-06-24</summary>
    Argument 2019-06-24

         - Epochs: 50
         - Batch size: 15
         - Learning rate: 0.0001 **(decay half at 60% & 80%)**
         - RNN size: 50
         - RNN layers: 4
         - **Input size: 21 frames**
         - **Output size: 1 frames**
         - **Weight: 20:1**
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-24/validall.png" alt="out1_decay_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-24/validpart.png" alt="out1_decay_part">
    </p>
</details>

---------------------------------------------------------------------------

Add the **Batch Normalize**, and **Dropout** to speed up training and strengthen the power of model.)**
Very slow (30M -> 2HR), Loss dicrease faster at the early stage (2HR)
<details>
    <summary>Add Batch normalize + Add Dropout 2019-06-25 (slow & bad)</summary>
    Argument 2019-06-25

         - Epochs: 50
         - Batch size: 15
         - Learning rate: 0.0001 **(decay half at 60% & 80%)**
         - RNN size: 50
         - RNN layers: 4
         - **Input size: 20 frames**
         - **Output size: 10 frames**
         - **Weight: 20:1**
         - **Batch normalize**
         - **Dropout: keep_rate = 0.9 (while training
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25/all.png" alt="DO_BN_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25/part.png" alt="DO_BN_part">
    </p>
</details>

The Batch Normalize make the loss decrease faster, but it took more than 3 times longer than the original method. So remove the BN only do **Dropout**.
<details>
    <summary>Only add Dropout 2019-06-25_2</summary>
    Argument 2019-06-25_2

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
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25_2/all.png" alt="DO_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-25_2/part.png" alt="DO_part">
    </p>
</details>

---------------------------------------------------------------------------

Expect the significant ups and downs wave, instead of high possility. Replace 3 different level of Gaussian Kernel, by apply a lower Gaussian Kernel 3 times.
<details>
    <summary>New Fine tune (replace 3 Gaussian Kernel by 1) 2019-06-30</summary>
    Argument 2019-06-30

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
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30/all.png" alt="new_kernal_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30/part.png" alt="new_kernal_part">
    </p>
</details>

Change validation data from ***run_front*** to ***squat***.
<details>
    <summary>New Fine tune (squat) 2019-06-30_squat</summary>
    Argument 2019-06-30_squat

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
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_squat/all.png" alt="new_kernal_all">
    </p>
</details>

Change the Batch size.
<details>
    <summary size="4">New Fine tune + 30 Batch size 2019-06-30_2 (squat) (Bad)</summary>
    Argument 2019-06-30_2

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
         - **Change validation data to *Squat***
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_2/all.png" alt="new_kernal_30_all">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_2/part.png" alt="new_kernal_30_part">
    </p>
</details>

---------------------------------------------------------------------------

We found that the validation loss did'nt show **Overfitting** correctly. Epoch 100 is showed as Overfitting, but its performace is better than lowest loss checkpoint.
<details>
    <summary>New Fine tune + 100 Epoch (squat) 2019-06-30_squat_100 (BEST)</summary>
    Argument 2019-06-30_squat_100

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
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-06-30_squat_100/all.png" alt="new_kernal_all">
    </p>
</details>

Try 100 epoch on original 3 diffrent level Gaussian Kernel. It become better, but **New fine tune** still better.
<details>
    <summary>Bring back 3 diffrent Gaussian Kernel + 100 Epoch 2019-07-01 (similar to 30 Batch size)</summary>
    Argument 2019-07-01

         - Epochs: **100**
         - Batch size: 15
         - Learning rate: 0.0001 (decay half at 60% & 80%)
         - RNN size: 50
         - RNN layers: 4
         - Input size: 20 frames
         - Output size: 10 frames
         - Weight: 20:1
         - Dropout: keep_rate = 0.9 (while training)
         - ~~**Cancel the 3 different Gaussian Kernel**~~
         - **Change validation data to *Squat***
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-07-01/all.png" alt="old_kernal_all">
    </p>
</details>

Try New Fine tune & 100 Epoch several times. Rewrite inference method.
<details>
    <summary>New Fine tune + 100 Epoch + New Inference 2019-07-01_2_3 (2nd best)</summary>
    Argument 2019-07-01_2_3

         - Epochs: 100
         - Batch size: 15
         - Learning rate: 0.0001 (decay half at 60% & 80%)
         - RNN size: 50
         - RNN layers: 4
         - Input size: 20 frames
         - Output size: 10 frames
         - Weight: 20:1
         - Dropout: keep_rate = 0.9 (while training)
         - Cancel the 3 different Gaussian Kernel
         - Change validation data to *Squat*
Validation result 
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-07-01_2/all.png" alt="Best_all2">
    </p>
    <p align="center" width="150">
        <img src="https://github.com/Jhuangsp/SegmentModel/blob/master/info/2019-07-01_3/all.png" alt="Best_all3">
    </p>
</details>

## Progress 2019-07-02
 - Good for model
    - Add decay
    - Add weighting
    - Output 10 instead 1
    - Dropout
    - 1 Kerel 3 times
    - More epoch regardless overfitting (Proper evaluation method)
 - Bad for model
    - Batch Normalize
    - Batch size (hyperopt)
    - too large Weighting scale

## TODO
 - 3 Method output
    - Probability: strengthen the non-zero field penalty (done)
    - Scale output: odd frames input, output 1 frame information (done)
    - ~~Time error~~
 - ~~Try DTW Discrete Time Warping/Dynamic Time Warping and hidden markov model (by Rabiner)~~
 - ~~Data preprocess~~
 - **Peak finding**
 - **Proper evaluation method**
 - Auto hyperparameter finding
 - Code review with handsomeguy

