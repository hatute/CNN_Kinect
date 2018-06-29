import Storage as ST
import Model as MD
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    path_list = [['./TrainingSamples/Ges_0', 0],
                 ['./TrainingSamples/Ges_1', 1],
                 ['./TrainingSamples/Ges_2', 2],
                 ['./TrainingSamples/Ges_3', 3],
                 ['./TrainingSamples/Ges_3-A', 6],
                 ['./TrainingSamples/Ges_3-B', 7],
                 ['./TrainingSamples/Ges_4', 4],
                 ['./TrainingSamples/Ges_5', 5]
                 ]

    file_list = ST.enum_samples(path_list)

    train_op, input_ph, label_ph, model = MD.train_operation(10, 192, 192, 3, 8)

    correct = 0.0
    total = 0.0

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for x in range(0, 2400):
            images, labels = ST.pick_some(file_list, 10)
            if (x+1) % 20 == 0:
                rlt = sess.run(model, feed_dict={input_ph: images})
                stat = np.argmax(rlt, axis=1) - np.argmax(labels, axis=1)
                stat = 10 - np.count_nonzero(stat)
                correct += stat
                total += 10
                rate = 100 * correct / total
                print(x+1, rate)
                # print(np.argmax(rlt, axis=1))
                # print(np.argmax(labels, axis=1))
            else:
                sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
                # print(x)

        MD.save_params('params_separate.bin', sess)