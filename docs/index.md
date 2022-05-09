# 3D Pose Estimation Using 2D Supervision

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

### What is the idea?


### Why does it make sense? How does it relate to prior work?


## Experiments

### Data

### Model Training


## Results


## Conclusion and Future Directions