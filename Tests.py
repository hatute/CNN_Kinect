import Storage as ST
import ModelTest as MDT
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ != '__main__':
    ss = np.ones([192, 192], np.float32)
    plt.imshow(ss, cmap='gray', vmin=0.0, vmax=1.0)
    plt.show()


if __name__ == '__main__':
    path_list = [['./Samples/Ges_0', 0],
                 ['./Samples/Ges_1', 1],
                 ['./Samples/Ges_2', 2],
                 ['./Samples/Ges_3', 3],
                 ['./Samples/Ges_3-A', 3],
                 ['./Samples/Ges_3-B', 3],
                 ['./Samples/Ges_4', 4],
                 ['./Samples/Ges_5', 5]
                 ]

    file_list = ST.enum_samples(path_list)

    evaluate_op, input_ph, c1, c2 = MDT.evaluate_opration('params_mixed.bin', 192, 192, 3, 6)

    correct = 0.0
    total = 0.0

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        image, label = ST.pick_some(file_list, 1)

        r1 = sess.run(c1, feed_dict={input_ph: image})
        img1 = np.zeros([192, 192], np.float32)

        # value ranges from 0.0 to 2.0

        for n in range(0, 8):
            for x in range(0, 192):
                for y in range(0, 192):
                    img1[x][y] = ((r1[0][x][y][n]) * 4)
            name = './_temp/c1_' + str(n) + '.png'
            plt.imsave(name, img1, cmap='gray', vmin=0.0, vmax=1.0)

        r2 = sess.run(c2, feed_dict={input_ph: image})
        img2 = np.zeros([96, 96], np.float32)
        for n in range(0, 32):
            for x in range(0, 96):
                for y in range(0, 96):
                    img2[x][y] = ((r2[0][x][y][n]) * 4)
            name = './_temp/c2_' + str(n) + '.png'
            plt.imsave(name, img2, cmap='gray', vmin=0.0, vmax=1.0)
