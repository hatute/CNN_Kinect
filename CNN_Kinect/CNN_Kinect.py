import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets(
    "./Python/TensorFlow/MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784], name='input_X')
y_ = tf.placeholder("float", shape=[None, 10], name='input_lable')
W = tf.Variable(tf.zeros([784, 10]), name='Weight')
b = tf.Variable(tf.zeros([10]), name='Bais')


def init_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Cont 1
W_conv1 = init_weight_variable([5, 5, 1, 32])  # 5*5 D1 32*featureMap_Weight
b_conv1 = init_bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Cont2
W_conv2 = init_weight_variable([5, 5, 32, 64])
b_conv2 = init_bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# link
W_fc1 = init_weight_variable([7 * 7 * 64, 1024])
b_fc1 = init_bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output
W_fc2 = init_weight_variable([1024, 10])
b_fc2 = init_bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
write = tf.summary.FileWriter("./Python/TensorFlow/graphs", sess.graph)
for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("epoch %d, training accuracy: %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

for i in range(5):
    testSet = mnist.test.next_batch(50)
    print("test %d, accuracy: %g" % (i, accuracy.eval(
        feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})))

write.close

# result:0.992

