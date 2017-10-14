import tensorflow as tf
import numpy as np
import os
N_CLASSES = 3
IMG_W = 128  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 128
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 600 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
# MINIST
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets (
#    "./MNIST_data/", one_hot=True)

class prepare(object):

    def __init__(self):
        self.sampleRootPath = "./SAMPLE/"
        self.samplePaperPath = "./SAMPLE/Paper/"
        self.sampleRockPath = "./SAMPLE/Rock/"
        self.sampleScissorsPath = "./SAMPLE/Scissors/"
        

    def createTargetAndLabel(self):
    #args:
    #    path: file directory
    #returns:
    #    list of images and labels
        paperTarget = []
        rockTarget = []
        scissorsTarget = []

        labelPaper = []    #2
        labelRock = []     #0
        labelScissors = [] #1

        for file in os.listdir(self.sampleRockPath):
            rockTarget.append(self.sampleRockPath + file)
            labelRock.append(int(0))
        for file in os.listdir(self.sampleScissorsPath):
            scissorsTarget.append(self.sampleScissorsPath + file)
            labelScissors.append(int(1))
        for file in os.listdir(self.samplePaperPath):
            paperTarget.append(self.samplePaperPath + file)
            labelPaper.append(int(2))

        print("SAMPLE STORAGE: \n %d Rock\n %d Scissors\n %d Paper\n" % (len(rockTarget),len(scissorsTarget),len(paperTarget)))

        sampleList = np.hstack((rockTarget,scissorsTarget,paperTarget))
        labelList = np.hstack((labelRock,labelScissors,labelPaper))

        return sampleList, labelList

    def createBatch(self, image, label, batch_size, capacity):
    #Args:
    #    image: list type
    #    label: list type
    #    batch_size: batch size
    #    capacity: the maximum elements in queue
    #Returns:
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
        image.set_shape([128,128,3])
        
        # RESIZE the sample pictures
        # image = tf.image.resize_image_with_crop_or_pad(image, image_W,
        # image_H)

        image = tf.image.per_image_standardization(image)

        # you can also use shuffle_batch
        image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                          batch_size=batch_size,
                                                          num_threads=64,
                                                          capacity=capacity,
                                                          min_after_dequeue=capacity - 1)

        label_batch = tf.reshape(label_batch, [batch_size])
        image_batch = tf.cast(image_batch, tf.float32)
    
        return image_batch, label_batch

class TFModel(object):

    def init_weight_variable(self,inputShape):
        weight = tf.get_variable('weights', 
                                      shape = inputShape,
                                      dtype = tf.float32, 
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        return weight

    def init_bias_variable(self,inputShape):
        bias = tf.get_variable('biases', 
                                     shape = inputShape,
                                     dtype = tf.float32,
                                     initializer = tf.constant_initializer(0.1))
        return bias

    def conv2d(self,input, weight):
        return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')


    def Cov(self,images, batch_size, n_classes):
        with tf.variable_scope('conv1') as scope:
            weights = self.init_weight_variable([3,3,3, 16])
            #weights = tf.get_variable('weights',
            #                          shape = [3,3,3, 16],
            #                          dtype = tf.float32,
            #                          initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            biases = self.init_bias_variable([16])
            #biases = tf.get_variable('biases', 
            #                         shape=[16],
            #                         dtype=tf.float32,
            #                         initializer=tf.constant_initializer(0.1))
            conv = self.conv2d(images , weights)
            #conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
            #pool1 and norm1
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                       padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                                  beta=0.75,name='norm1')
    
        #conv2
        with tf.variable_scope('conv2') as scope:
            weights = self.init_weight_variable([3,3,16, 16])
            #weights = tf.get_variable('weights',
            #                          shape=[3,3,16,16],
            #                          dtype=tf.float32,
            #                          initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            biases = self.init_bias_variable([16])
            #biases = tf.get_variable('biases',
            #                         shape=[16], 
            #                         dtype=tf.float32,
            #                         initializer=tf.constant_initializer(0.1))
            conv = self.conv2d(norm1 , weights)
            #conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
        #pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75,name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                                   padding='SAME',name='pooling2')
    
    
        #local3
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                      shape=[dim,128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[128],
                                     dtype=tf.float32, 
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)    
    
        #local4
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                      shape=[128,128],
                                      dtype=tf.float32, 
                                      initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
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
                                      initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
            biases = tf.get_variable('biases', 
                                     shape=[n_classes],
                                     dtype=tf.float32, 
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
        return softmax_linear

    def losses(self,logits, labels):
        #Compute loss from logits and labels
        #Args:
        #    logits: logits tensor, float, [batch_size, n_classes]
        #    labels: label tensor, tf.int32, [batch_size]
        
        #Returns:
        #    loss tensor of float type
    
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                            (logits=logits, labels=labels, name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name+'/loss', loss)
        return loss

    def trainning(self,loss, learning_rate):
        #Training ops, the Op returned by this function is what must be passed to 
        #    'sess.run()' call to cause the model to train.
        
        #Args:
        #    loss: loss tensor, from losses()
        
        #Returns:
        #    train_op: The op for trainning
        
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step= global_step)
        return train_op

    def evaluation(self,logits, labels):
         #Evaluate the quality of the logits at predicting the label.
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
          tf.summary.scalar(scope.name+'/accuracy', accuracy)
      return accuracy

class TFRun():

    def run_training(self):
    
        # you need to change the directories to yours.
        #train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
        logs_train_dir = './logs/train/'
    
        #train, train_label = input_data.get_files(train_dir)
        P = prepare()
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
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver()
    
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
            
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
        coord.join(threads)
        sess.close()


run = TFRun()
run.run_training()
































# OFFICIAL SAMPLE

#sess = tf.InteractiveSession()

#x = tf.placeholder("float", shape=[None, 784], name='input_X')
#y_ = tf.placeholder("float", shape=[None, 10], name='input_lable')
#W = tf.Variable(tf.zeros([784, 10]), name='Weight')
#b = tf.Variable(tf.zeros([10]), name='Bais')


#def init_weight_variable(shape):
#    initial = tf.truncated_normal(shape, stddev=0.1)
#    return tf.Variable(initial)


#def init_bias_variable(shape):
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial)

#def conv2d(x, W):
#    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#def max_pool_2x2(x):
#    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                          strides=[1, 2, 2, 1], padding='SAME')


## Cont 1
#W_conv1 = init_weight_variable([5, 5, 1, 32])  # 5*5 D1 32*featureMap_Weight
#b_conv1 = init_bias_variable([32])
#x_image = tf.reshape(x, [-1, 28, 28, 1])
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

## Cont2
#W_conv2 = init_weight_variable([5, 5, 32, 64])
#b_conv2 = init_bias_variable([64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

## link
#W_fc1 = init_weight_variable([7 * 7 * 64, 1024])
#b_fc1 = init_bias_variable([1024])
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout
#keep_prob = tf.placeholder("float")
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Output
#W_fc2 = init_weight_variable([1024, 10])
#b_fc2 = init_bias_variable([10])
#y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

## Evaluation
#correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#sess.run(tf.global_variables_initializer())
#write = tf.summary.FileWriter("./Python/TensorFlow/graphs", sess.graph)
#for i in range(5000):
#    batch = mnist.train.next_batch(50)
#    if i % 1000 == 0:
#        train_accuracy = accuracy.eval(feed_dict={
#            x: batch[0], y_: batch[1], keep_prob: 1.0})
#        print("epoch %d, training accuracy: %g" % (i, train_accuracy))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#for i in range(5):
#    testSet = mnist.test.next_batch(50)
#    print("test %d, accuracy: %g" % (i, accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})))

#write.close

## result:0.992

