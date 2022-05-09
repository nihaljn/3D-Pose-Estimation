# 3D Pose Estimation Using 2D Supervision

|Nihal Jain|Talha Faiz|Titas Chakraborty|
|:---:|:---:|:---:|
|nihalj@andrew.cmu.edu|||

<!-- 
Motivation: Why is it important and relevant? Why should we care?
Prior Work: Briefly mention related works that are relevant to your idea instead of covering all related works.
Your idea: What is the idea?
Your idea: Why does your idea make sense intuitively?
Your idea: How does it relate to prior work in the area?
Your results: If it worked, how much did you improve?
Your results: If it did not work, why did you expect it to work?
Your results: Any negative results? Maybe you figured after trying that it did not make sense.
Summary and Conclusion
Future directions 
-->

## Introduction and Motivation

3D Pose Estimation is an important research topic with numerous applications in fields such as computer animation and action recognition. The general problem framework for 3D Pose Estimation consists of a single 2D image or a sequence of 2D images representing one or more humans as input to our model. Our model outputs one 3D body represention per 2D image representing the pose of the human in that image. A common representation for a 3D person is the 3D locations of the body joints.

Several methods exist to detect these joints in the 2D images. If we can detect these keypoints in 2D, we only need to translate them to 3D to obtain the 3D pose. Recent work has shown that even a simple neural network model trained on 2D keypoints can achieve acceptable results on this problem. Some approaches have exploited temporality using the fact that in many applications, the 2D images form a video. Other approaches have tried to improve keypoint detection by taking into account occlusion.

However, most methods perform monocular reconstruction. They assume that only one camera captures the 2D images of a person. However, in most applications, it is simple to place another camera to capture 2D images of the person from a different angle. A multi-view camera setup is something we can exploit to obtain better 3D pose estimation. However, most methods perform supervised reconstruction. They assume access to ground truth 3D poses which is difficult to obtain in most practical settings. To solve that problem, we propose a 3D pose estimation framework which relies only on 2D supervision and does not assume access to 3D ground truth labels. Our results showcase that our model, trained using multi-view camera images is competitive with 3D supervised methods using single-view images at test time. If we assume multi-view images at test time, our method performs much better than 3D supervised methods. 


## Related Work


## Idea

In this section, we go over the details of our method and how it relates to prior work in this area.

### What is the idea?

Our approach is based on the simple idea that multiple 2D views can be obtained from a single 3D view. Concretely, if we estimate some 3D pose of a frame, we should be able to rotate and project that 3D pose onto different views and obtain consistent 2D poses. Our model is trained using a loss function that captures this intuition. We next describe the details of our method in detail.

#### *Assumptions*

We make the following modeling assumptions in our approach:
- <strong>Our model lifts 2D poses to 3D</strong>: We assume access to 2D poses from 2D images. These are easy to obtain and recent work has achieved very high levels of performance in estimating 2D poses from 2D images [].
- <strong>Train-time Data</strong>: We assume access to multiple views of the same frame during training. Further, we also assume access to the camera intrinsic and extrinsic parameters used to capture these different scenes at training time. In our experiments, we try two approaches: (1) takes a single frame as input at a time, (2) takes a sequence of frames as input at a time (assuming access to video-feed of poses).
- <strong>Test-time Data</strong>: We assume access to single views of the same frame during testing and no camera parameters. Further, we show in our experiments, that if indeed we have access to multiple views and camera parameters at test time, we can achieve enhanced performance on the test data.

#### *Model*

<img src="https://raw.githubusercontent.com/nihaljn/3D-Pose-Estimation/site/docs/files/model_diagram.png?token=GHSAT0AAAAAABQ3ZLRNTQYZSMXUAWO4OH46YTZNP3Q"></img>
<em><strong>Figure []. Model Architecture.</strong> This diagram shows our simple model architecture taken from []. We take 2D poses with respect to a camera as input and estimate 3D poses as output with respect to the same camera.</em>

Figure [] shows our model with its basic building blocks. The model design is such that it takes 2D poses as inputs and produces 3D poses as outputs. This approach, adapted from [], is based on a simple fully-connected neural network with batch normalization, dropout, Rectified Linear Units (RELUs), and residual connections []. There are two further layers: one to increase dimensionality to 512 just before the input to the model in the diagram, and one that projects the output of the model to get 3D poses. In our experiments we use 2 residual blocks, so we have a total of 4 linear layers.

Having obtained estimates of 3D poses using this model, we can measure the quality of the estimates using the Mean Per-Joint Position Error (MPJPE), which is a popular metric in 3D pose estimation literature, and often referred to as protocol #1 []. MPJPE measure the average euclidean distance between the predicted and actual coordinates of each joint in the dataset. Mathematically,

<img src="https://render.githubusercontent.com/render/math?math={\text{MPJPE} = \displaystyle \frac{1}{N*J} \sum_{i = 1}^N \sum_{j = 1}^J || y_i^j - \hat{y}_i^j ||}">

where, <img src="https://render.githubusercontent.com/render/math?math={N, J}"> are the number of examples and joints respectively, and <img src="https://render.githubusercontent.com/render/math?math={\hat{y}, y}"> are the estimated pose and ground truth pose respectively. Note that the same formulation holds for poses in 3D or 2D.

We train this model using two approaches:
<strong>1. Baseline.</strong> 
<strong>2. Without labels (ours).</strong>

### Why does it make sense? How does it relate to prior work?


## Experiments

### Data

### Model Training


## Results


## Conclusion and Future Directions

## References

