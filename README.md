# Digital Creativity
## Audio to Image with pix2pix

### How to use
In order to run and play with the model, the required Touchdesigner file has to be opened.
### Create Project
#### Prerequisite
For this project an Nvidia GPU with CUDA is needed.
Download <a href="https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64">CUDA here</a>
or use pix2pix <a href="https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb
">Colab Notebook.</a>

#### Preparing the project
Download the pix2pix model from <a href="https://github.com/ML-and-AI-repo/pytorch-CycleGAN-and-pix2pix">junyanz repository</a> or again use the <a href="https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb
">Colab Notebook.</a> <br />
In this project the pytorch model will be used. <br />
After downloading the pix2pix model, a specific library is needed for the test later. Download the spout library <a href="https://github.com/Ajasra/Spout-for-Python">here.</a> 


#### Used packages
For the package management I used both Anaconda and pip.
Here is the list that are needed for this project:
<ul>
  <li>!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117</li>
  <li>!pip install dominate</li>
  <li>!pip install pygame</li>
  <li>!pip install pyopengl</li>
</ul>

**Important**: It is important to notice, that the extra-index-url depends on which CUDA-Version you currently have installed.

#### Setup the project
After preparing the 

### Train
#### Create Datasets

### Test
#### Touchdesigner

### Sources and Thanks
Idea, Help and Sources where found <a href="https://medium.com/@vasily.onl/visualizing-sound-with-ai-e7a9191fea2c">here</a> thanks to Vasily Batin.
