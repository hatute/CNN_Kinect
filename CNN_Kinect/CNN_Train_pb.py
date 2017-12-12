import os
import time

import numpy as np
import tensorflow as tf

halfSize = 96
fullSize = 192

N_CLASSES = 3
IMG_W = 128  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 128
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 100  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001
logs_dir = './logs_pb/'


class Prepare(object):
    def __init__(self):
        self.sampleRootPath = "./SAMPLE/"
        self.samplePaperPath = "./SAMPLE/Paper/"
        self.sampleRockPath = "./SAMPLE/Rock/"
        self.sampleScissorsPath = "./SAMPLE/Scissors/"

    def createTargetAndLabel(self):
        # args:
        #    path: file directory
        # returns:
        #    list of images and labels
        paperTarget = []
        rockTarget = []
        scissorsTarget = []

        labelPaper = []  # 2
        labelRock = []  # 0
        labelScissors = []  # 1

        for file in os.listdir(self.sampleRockPath):
            rockTarget.append(self.sampleRockPath + file)
            labelRock.append(int(0))
        for file in os.listdir(self.sampleScissorsPath):
            scissorsTarget.append(self.sampleScissorsPath + file)
            labelScissors.append(int(1))
        for file in os.listdir(self.samplePaperPath):
            paperTarget.append(self.samplePaperPath + file)
            labelPaper.append(int(2))

        print("SAMPLE STORAGE: \n %d Rock\n %d Scissors\n %d Paper\n" % (
            len(rockTarget), len(scissorsTarget), len(paperTarget)))

        sampleList = np.hstack((rockTarget, scissorsTarget, paperTarget))
        labelList = np.hstack((labelRock, labelScissors, labelPaper))

        temp = np.array([sampleList, labelList])
        temp = temp.transpose()
        np.random.shuffle(temp)

        sampleList = list(temp[:, 0])
        labelList = list(temp[:, 1])
        labelList = [int(i) for i in labelList]

        return sampleList, labelList

    def createBatch(self, image, label, batch_size, capacity):
        # Args:
        #    image: list type
        #    label: list type
        #    batch_size: batch size
        #    capacity: the maximum elements in queue
        # Returns:
        #    image_batch: 4D tensor [batch_size, width, height, 3],
        #    dtype=tf.float32
        #    label_batch: 1D tensor [batch_size], dtype=tf.int32

        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int32)

        # make an input queue
        input_queue = tf.train.slice_input_producer([image, label])

        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image_contents, channels=3)
        image.set_shape([fullSize, fullSize, 3])

        # RESIZE the sample pictures
        # image = tf.image.resize_image_with_crop_or_pad(image, image_W,
        # image_H)

        image = tf.image.per_image_standardization(image)

        # you can also use shuffle_batch
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=64,
                                                          capacity=capacity,
                                                          min_after_dequeue=capacity - 1)

        label_batch = tf.reshape(label_batch, [batch_size])
        image_batch = tf.cast(image_batch, tf.float32)

        return image_batch, label_batch


class TFModel(object):
    def Cov(self, images, batch_size, n_classes):
        with tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 3, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                   padding='SAME', name='pooling2')

        # local3
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                      shape=[dim, 128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            # local4
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                      shape=[128, 128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[128, n_classes],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[n_classes],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

        return softmax_linear

    def losses(self, logits, labels):
        # Compute loss from logits and labels
        # Args:
        #    logits: logits tensor, float, [batch_size, n_classes]
        #    labels: label tensor, tf.int32, [batch_size]

        # Returns:
        #    loss tensor of float type

        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                (logits=logits, labels=labels, name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
        return loss

    def trainning(self, loss, learning_rate):
        # Training ops, the Op returned by this function is what must be passed to
        #    'sess.run()' call to cause the model to train.

        # Args:
        #    loss: loss tensor, from losses()

        # Returns:
        #    train_op: The op for trainning

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        # Evaluate the quality of the logits at predicting the label.
        # Args:
        #   logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        #   labels: Labels tensor, int32 - [batch_size], with values in the
        #     range [0, NUM_CLASSES).
        # Returns:
        #   A scalar int32 tensor with the number of examples (out of batch_size)
        #   that were predicted correctly.

        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy


class TFRun(object):
    def runTraining(self):
        # train, train_label = input_data.get_files(train_dir)
        P = Prepare()
        M = TFModel()
        train, train_label = P.createTargetAndLabel()

        train_batch, train_label_batch = P.createBatch(train,
                                                       train_label,
                                                       BATCH_SIZE,
                                                       CAPACITY)
        train_logits = M.Cov(train_batch, BATCH_SIZE, N_CLASSES)
        train_loss = M.losses(train_logits, train_label_batch)
        train_op = M.trainning(train_loss, learning_rate)
        train__acc = M.evaluation(train_logits, train_label_batch)

        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            for step in np.arange(MAX_STEP + 1):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

                if step % 100 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))

                    # ckpt save way
                    #     summary_str = sess.run(summary_op)f
                    #     train_writer.add_summary(summary_str, step)

                    # if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    #     checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    # saver.save(sess, logs_dir, global_step=step)

                    constant_graph = tf.get_default_graph().as_graph_def()
                    current_time = str(time.strftime('%Y%m%d', time.localtime(time.time())))
                    file_name = current_time + "_" + "step" + str(step) + ".pb"
                    with tf.gfile.FastGFile(logs_dir + file_name, mode='wb') as f:
                        f.write(constant_graph.SerializeToString())

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    run = TFRun()
    run.runTraining()
