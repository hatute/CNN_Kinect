from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import tensorflow as tf
import numpy as np
import os
import ctypes
import _ctypes
import pygame
import sys
import CNN_Kinect.CNN_TF
from PIL import Image

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


class BodyFrameRuntime(object):
    def __init__(self):
        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
        pygame.display.set_caption("Gesture Recgnization by Kinect")
        # Loop until the user clicks the close button.
        self._done = False
        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface(
            (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        # here we will store skeleton data
        self._bodies = None

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def extract_hand(self, jointPoints):
        rightHand_joint = jointPoints[PyKinectV2.JointType_HandRight]
        rightHand_centerPoint = (int(rightHand_joint.x), int(rightHand_joint.y))
        rect_topLeft = (rightHand_centerPoint[0] - 64, rightHand_centerPoint[1] - 64)
        rect_bottomRight = (rightHand_centerPoint[0] + 64, rightHand_centerPoint[1] + 64)

        seclectedRect = pygame.Rect(rightHand_centerPoint[0] - 64, rightHand_centerPoint[1] - 64, 128, 128)

        # UNCOMMENT IF NEED TO DRAW THE CIRCLE FROM RIGHTHAND
        pygame.draw.circle(self._frame_surface, pygame.color.THECOLORS["blue"], rightHand_centerPoint, 80, 4)

        return self._frame_surface.subsurface(seclectedRect)

    def runEvaluate(self):
        # -------- Main Program Loop -----------
        iter = 0

        while not self._done:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop
                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # get skeletons
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # draw skeletons to _frame_surface
            subSurface_handPart = None
            if self._bodies is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    joints = body.joints
                    # convert joint coordinates to color space
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    subSurface_handPart = self.extract_hand(joint_points)
                    # self._frame_surface.blit(subSurface, (0,0))
                    # self._screen.blit(subSurface, (0,0))
                    # self.draw_body(joints, joint_points, SKELETON_COLORS[i])

            ratio_heightToWidth = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(ratio_heightToWidth * self._screen.get_width())

            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height))
            self._screen.blit(surface_to_draw, (0, 0))
            # surface_to_draw = None
            if subSurface_handPart is not None:
                self._screen.blit(subSurface_handPart, (0, 0))

            pygame.display.update()

            # update the screen with what we've drawn.
            pygame.display.flip()
            # Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()





if __name__ == "__main__":
    Running = BodyFrameRuntime()
    with tf.Session as sessMain:
        image = tf.zeros([1, 128, 128, 3])
        model = CNN_Kinect.CNN_TF.TFModel()
        logit = model.Cov(image, 1, 3)
        logit = tf.nn.softmax(logit)
        x = tf.placeholder(tf.float32, shape=[128, 128, 3])
        LOGDIR = "./log/train/"
        saver = tf.train.Saver()
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(LOGDIR)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sessMain, ckpt.model_checkpoint_path)
            print('Loading success, Latest global_step is %s' % global_step)
        else:
            print('No checkpoint file found')


        # TODO: get image resource
    Running.runEvaluate()
