
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

import sys
sys.path.append('Library')

import numpy as np
import argparse
import time
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *

import torch

""" here your functions """
def main_pipeline(data, model, dataset):
    
    data = (data / 255.0) * 2 - 1
    data = np.transpose(data, [2, 0, 1])
    data_torch = torch.Tensor([data])

 
    dataset['A'] = data_torch
    model.set_input(dataset)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results

    output = visuals['fake_B'].data.cpu().numpy()
    output = np.transpose(output, [0,2,3,1])[0] #change order of the output

    #convert texture
    output = (output + 1) * 255 / 2

    return output


def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for i, data in enumerate(dataset):
        if i >= 1: 
            break
        defdata = data

    # window details
    width = 512 
    height = 512 
    display = (width,height)
    
    req_type = 'input-output'
    receiverName = 'input'
    senderName = 'output'
    silent = True
    
    # window setup
    pygame.init() 
    pygame.display.set_caption(senderName)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)

    if req_type == 'input' or req_type == 'input-output':
        # init spout receiver
        spoutReceiverWidth = width
        spoutReceiverHeight = height
        # create spout receiver
        spoutReceiver = SpoutSDK.SpoutReceiver()
	    # Its signature in c++ looks like this: bool pyCreateReceiver(const char* theName, unsigned int theWidth, unsigned int theHeight, bool bUseActive);
        spoutReceiver.pyCreateReceiver(receiverName,spoutReceiverWidth,spoutReceiverHeight, False)
        # create textures for spout receiver and spout sender 
        textureReceiveID = glGenTextures(1)
        
        # initalise receiver texture
        glBindTexture(GL_TEXTURE_2D, textureReceiveID)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # copy data into texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, spoutReceiverWidth, spoutReceiverHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None ) 
        glBindTexture(GL_TEXTURE_2D, 0)

    if req_type == 'output' or req_type == 'input-output':
        # init spout sender
        spoutSender = SpoutSDK.SpoutSender()
        spoutSenderWidth = width
        spoutSenderHeight = height
	    # Its signature in c++ looks like this: bool CreateSender(const char *Sendername, unsigned int width, unsigned int height, DWORD dwFormat = 0);
        spoutSender.CreateSender(senderName, spoutSenderWidth, spoutSenderHeight, 0)
        # create textures for spout receiver and spout sender 
    textureSendID = glGenTextures(1)

    # loop for graph frame by frame
    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                spoutReceiver.ReleaseReceiver()
                pygame.quit()
                quit()
        
        if req_type == 'input' or req_type == 'input-output':
            # receive texture
            # Its signature in c++ looks like this: bool pyReceiveTexture(const char* theName, unsigned int theWidth, unsigned int theHeight, GLuint TextureID, GLuint TextureTarget, bool bInvert, GLuint HostFBO);
            if sys.version_info[1] == 5:
                spoutReceiver.pyReceiveTexture(receiverName, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID, GL_TEXTURE_2D, False, 0)
            else:
                spoutReceiver.pyReceiveTexture(receiverName, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID.item(), GL_TEXTURE_2D, False, 0)

            glBindTexture(GL_TEXTURE_2D, textureReceiveID)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            # copy pixel byte array from received texture   
            data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)  #Using GL_RGB can use GL_RGBA 
            glBindTexture(GL_TEXTURE_2D, 0)
            # swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
            data.shape = (data.shape[1], data.shape[0], data.shape[2])
        else:
            data = np.ones((width,height,3))*255
        
        # call our main function
        output = main_pipeline(data, model, defdata)
        
        # setup the texture so we can load the output into it
        glBindTexture(GL_TEXTURE_2D, textureSendID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # copy output into texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, output )
            
        # setup window to draw to screen
        glActiveTexture(GL_TEXTURE0)
        # clean start
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
        # reset drawing perspective
        glLoadIdentity()
        # draw texture on screen
        glBegin(GL_QUADS)

        glTexCoord(0,0)        
        glVertex2f(0,0)

        glTexCoord(1,0)
        glVertex2f(width,0)

        glTexCoord(1,1)
        glVertex2f(width,height)

        glTexCoord(0,1)
        glVertex2f(0,height)

        glEnd()
        
        if silent:
            pygame.display.iconify()
                
        # update window
        pygame.display.flip()        

        if req_type == 'output' or req_type == 'input-output':
            # Send texture to spout...
            # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);
            if sys.version_info[1] == 5:
                spoutSender.SendTexture(textureSendID, GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)
            else:
                spoutSender.SendTexture(textureSendID.item(), GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)

        
        
if __name__ == '__main__':
    main()
