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

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                    pygame.color.THECOLORS["blue"],
                    pygame.color.THECOLORS["green"],
                    pygame.color.THECOLORS["orange"], 
                    pygame.color.THECOLORS["purple"], 
                    pygame.color.THECOLORS["yellow"], 
                    pygame.color.THECOLORS["violet"]]


class ReaderDepthCsv(object):

    def __init__(self, filename):
        self._filename = filename
        #self._allFrames = np.genfromtxt(filename, delimiter=',')

        with h5py.File(filename, 'r') as hf:
            self._allFrames = hf['all-frames'][:]

        self._height = 424
        self._chestDistances = []
        self._y1 = None
        self._y2 = None
        self._x1 = None
        self._x2 = None


    def select_roi(self, indexFrame):
        img_array = self._allFrames[indexFrame].reshape((self._height, -1))
        plt.imsave('outfile.png', img_array)

        # TODO delete file

        z_img = cv2.imread('outfile.png')

        r = cv2.selectROI(z_img, fromCenter=0)
        self._y1 = int(r[1])
        self._y2 = int(r[3] + r[1])
        self._x1 = int(r[0])
        self._x2 = int(r[0] + r[2])

        cv2.imshow('Image', img_array[self._y1:self._y2, self._x1:self._x2])
        cv2.waitKey()



    def show_depth_plot(self, t_stamp):

        for index_frame, frame in enumerate(self._allFrames):
            img_reshaped = frame.reshape((self._height, -1))
            frame_cropped = img_reshaped[self._y1:self._y2, self._x1:self._x2]
            self._chestDistances.append(np.mean(frame_cropped))


        plt.plot(np.arange(0, len(self._chestDistances) * t_stamp, t_stamp), self._chestDistances,'m-')
        plt.xlabel("czas [s]")
        plt.ylabel("odległość [mm]")
        plt.show(block=True)


class DepthRuntime(object):
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

        pygame.display.set_caption("Kinect for Windows v2 Depth")



    def draw_depth_frame(self, frame, target_surface):
        if frame is None:
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def save(self, filename):
        # Saving to a CSV file
        #np.savetxt(filename, self._allFrames, delimiter=",")

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("all-frames", data=self._allFrames)

    def run(self):
        # -------- Main Program Loop -----------
        plt.figure(10)
        plt.ion()

        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            # --- Getting frames and drawing  
            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                self._allFrames.append(frame)
                self.draw_depth_frame(frame, self._frame_surface)
                frame = None

            self._screen.blit(self._frame_surface, (0,0))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            # self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Depth"



# Zapsujemy
shell = DepthRuntime()
shell.run()

#game.save('filename.h5')

# TODO podgląd?

# Odczytujemy
#reader = ReaderDepthCsv('filename.h5')
#reader.select_roi(0)
#reader.show_depth_plot(0.04)

