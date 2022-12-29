# Digital Creativity
## Audio to Image with pix2pix
By Michael Urlacher

### How to use
In order to run and play with the model, the required Touchdesigner file has to be opened. Open the downloaded *test2.toe* - it only has to be opened, but in the far left, you can change the Audio Input to test different audio.
<img src="imgs/TouchAudioInput.png" />
<br />
Click on either one of the two boxes and select you input device or audio file on the left side of the screen.
<img src="imgs/TouchAudioInput2.png" />
<br />
In the Project you can find *test2.bat* which starts the model. The Model is trained on the Doom-Images. <br />
Now, depending on the Audio Input, different Images should appear in *test2.toe*. If they don't on the far right side of *test2.toe* there is a box called "null" and next to the name is a circle to activate the display. <br />
<img src="imgs/TouchImageOutput.png" />
<br />
If the image is not changing, in the post processing part there is a box called *math*. There you can control the multiplier of the Audio, make sure its above **0**.
<img src="imgs/TouchImageOutput2.png" />
<br />


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
With conda, go to the root folder of the project and create a conda environment named *pix2pix* with Python 3.7.
```
conda create -n pix2pix python=3.7
```
Activate the environment:
```
activate pix2pix
```
and install all of the previous listed packages.
<br />
<br />
Two major things left to do. <br />
1. Finalise the setup by creating a new test.
2. Create Dataset and train the model.
First create a new folder in the project and name it "Library". In there insert *Spout.py*, *SpoutSDK-35.pyd*, *SpoutSDK-36.pyd* and *SpoutSDK.pyd*.
It should look like this: <br />
<img src="imgs/LibraryFolder.png" />
<br />
- Copy paste *test.py* and rename it. Mine is called *test2.py*
- Create a .bat file and call it the same.
<br />
In the .bat file copy paste this and change parts to your fits.
```
python copied-py-file.py --dataroot ./datasets/folder-name --name name-of-results --model pix2pix --direction AtoB --load_size 512 --crop_size 512
```
Below */datasets/folder-name* make sure to add another folder named *test* and or *train* and insert your dataset in there.


### Train
#### Create Datasets

### Test
#### Touchdesigner

### Sources and Thanks
Idea, Help and Sources where found <a href="https://medium.com/@vasily.onl/visualizing-sound-with-ai-e7a9191fea2c">here</a> thanks to Vasily Batin. <br />
Helpful solutions to problems:
- <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/commit/9bcef69d5b39385d18afad3d5a839a02ae0b43e7">Attribute Error</a>
- <a href="https://github.com/python-pillow/Pillow/issues/4130#issuecomment-1201516416">Import Error<a/>
