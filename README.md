# 3D Pose Estimation Using 2D Supervision

<i>Note: This is project work done towards the completion of [16-824 Visual Learning and Recognition](https://visual-learning.cs.cmu.edu/s22/index.html). Supporting detailed report available [here](https://nihaljn.github.io/3D-Pose-Estimation). </i>

3D Pose Estimation is an important research topic with numerous applications in fields such as computer animation and action recognition. The general problem framework for 3D Pose Estimation consists of a single 2D image or a sequence of 2D images representing one or more humans as input to a model. The model outputs one 3D body represention per 2D image representing the pose of the human in that image. A common representation for a 3D person is the set of 3D coordinates of the body joints, which the model can learn to output.

In this project, we propose a 3D pose estimation framework which relies only on 2D supervision and does not assume access to 3D ground truth labels. Our results showcase that our model, trained using multi-view camera images is competitive with 3D supervised methods using single-view images at test time. If we assume multi-view images at test time, our method performs much better than 3D supervised methods on the specific examples of interest. Figure below shows an illustration of our expected inputs and outputs.

<i>Note: The code in this repository was written hastily towards a course project deadline; it was not henceforth maintained. Nevertheless, I believe the code is runnable and results reproducible. If something appears broken or non-intuitive, please open an issue.</i>

## Data



## Usage

To run the code, modify `main.py` by putting the required options inside `Args`. Then simply run `python main.py`. This will run each file with the default settings used to produce the reported results.
