from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import ctypes
import pygame
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import h5py

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


class ReaderDepthCsv(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        
        # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height),
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        self._allFrames = []

        pygame.display.set_caption("Depth Frame demo 1")
    def run(self):
        while not self._done:
                    # --- Main event loop
                    for event in pygame.event.get(): # User did something
                        if event.type == pygame.QUIT: # If user clicked close
                            self._done = True # Flag that we are done so we exit this loop

                        elif event.type == pygame.VIDEORESIZE: # window resized
                            self._screen = pygame.display.set_mode(event.dict['size'], 
                                                        pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

                    if self._kinect.has_new_depth_frame():
                        wholeBodyFrame = self._kinect.get_last_depth_frame()
                        while wholeBodyFrame:
                            handFrame = self._fram
