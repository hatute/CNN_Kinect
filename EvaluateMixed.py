import Storage as ST
import Model as MD
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    path_list = [['./TrainingSamples/Ges_0', 0],
                 ['./TrainingSamples/Ges_1', 1],
                 ['./TrainingSamples/Ges_2', 2],
                 ['./TrainingSamples/Ges_3', 3],
                 ['./TrainingSamples/Ges_3-A', 3],
                 ['./TrainingSamples/Ges_3-B', 3],
                 ['./TrainingSamples/Ges_4', 4],
                 ['./TrainingSamples/Ges_5', 5]
                 ]

    file_list = ST.enum_samples(path_list)

    evaluate_op, input_ph = MD.evaluate_opration('params_mixed.bin', 192, 192, 3, 6)

    correct = 0.0
    total = 0.0

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for x in range(0, 1000):
            image, label = ST.pick_some(file_list, 1)
            result = sess.run(evaluate_op, feed_dict={input_ph: image})
            if np.argmax(result) == np.argmax(label):
                correct += 1
            total += 1
            print(correct / total)


