from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from PIL import Image
import matplotlib.pyplot as plt
import ctypes
import _ctypes
import pygame
import sys
import os
import numpy as np
import tensorflow as tf
import CNN_Kinect.CNN_TF

halfSize = 96
fullSize = 192

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread



file = ".\_tmp.png"


def evaluate_one_image(surface):
    image_array = np.array(Image.open(file))
    image_array = image_array.resize([192, 192])
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, fullSize, fullSize, 3])

        model = CNN_Kinect.CNN_TF.TFModel()

        logit = model.Cov(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[fullSize, fullSize, 3])

        # you need to change the directories to yours.
        logs_train_dir = './logs/train/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            result = ["Rock", "Sciosor", "Paper"]
            print(result[max_index])
            '''
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
                '''


class BodyGameRuntime(object):
    def __init__(self):

        pygame.init()
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
        pygame.display.set_caption("Gesture Recgnization by Kinect")
        # Loop until the user clicks the close button.
        self._done = False
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
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

    def extract(self, jointPoints):
        rightHand_joint = jointPoints[PyKinectV2.JointType_HandRight]
        rightHand_centerPoint = (int(rightHand_joint.x), int(rightHand_joint.y))
        rect_topLeft = (rightHand_centerPoint[0] - halfSize, rightHand_centerPoint[1] - halfSize)
        rect_bottomRight = (rightHand_centerPoint[0] + halfSize, rightHand_centerPoint[1] + halfSize)
        seclectedRect = pygame.Rect(rightHand_centerPoint[0] - halfSize, rightHand_centerPoint[1] - halfSize, fullSize,
                                    fullSize)
        # UNCOMMENT IF WANT TO DRAW THE CIRCLE FROM RIGHTHAND
        # pygame.draw.circle(self._frame_surface, pygame.color.THECOLORS["blue"], center, 80, 4)
        pygame.draw.rect(self._frame_surface, pygame.color.THECOLORS["blue"], seclectedRect, 4)

        return self._frame_surface.subsurface(seclectedRect)

    def run(self):

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

            # fill out back buffer surface with frame's data
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
                    subSurface_handPart = self.extract(joint_points)

                    # self._frame_surface.blit(subSurface, (0,0))
                    # self._screen.blit(subSurface, (0,0))
                    # self.draw_body(joints, joint_points, SKELETON_COLORS[i])


            # screen size may be different from Kinect's color frame size
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height))
            self._screen.blit(surface_to_draw, (0, 0))
            surface_to_draw = None
            if subSurface_handPart is not None:
                if iter % 30 == 0:
                    pygame.image.save(subSurface_handPart, file)
                    evaluate_one_image(subSurface_handPart)
                    os.remove(file)
                    # todo: 

            pygame.display.update()
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            # --- Limit to 60 frames per second
            self._clock.tick(60)
        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Frame"
game = BodyGameRuntime()
game.run()
