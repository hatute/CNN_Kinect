import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logs_dir = './logs_pb/'
test_image_path = './SAMPLE/Paper/'
halfSize = 96
fullSize = 192
batch_size = 16


def get_lastest_pbfile(path):
    lastest_pbfile = ""
    for target in os.listdir(path):
        file = str(path + target)
        if os.path.isfile(file):
            create_time = 0
            if (os.path.splitext(file)[-1] == '.pb') and (create_time < os.stat(file).st_ctime):
                create_time = os.stat(file).st_ctime
                lastest_pbfile = file
    return lastest_pbfile


def get_images_and_label(path, required_number, required_format):
    num = 0
    image_batch = []
    label_batch = []
    for target in os.listdir(path):
        if num < required_number:
            file = str(path + target)
            if os.path.splitext(file)[-1] == required_format:
                image_batch.append(file)
            else:
                print("error")
                quit()
            if 'Rock' in path:
                label_batch.append(0)
            if 'Scissors' in path:
                label_batch.append(1)
            if 'Paper' in path:
                label_batch.append(2)
            num += 1
    #         print("1")
    # print("2")
    return image_batch, label_batch


def eval_single_picture(pb_file_path, input_image_path):
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     with open(pb_file_path, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         image_array = np.array(Image.open(input_image_path))
    #         print(image_array)
    #         image = tf.image.per_image_standardization(image_array)
    #         print(image)
    #         output = tf.import_graph_def(graph_def, input_map={'input:0': image},
    #                                      return_elements=['softmax_linear/softmax_output:0'])
    #         print(sess.run(output))
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            out_softmax = sess.graph.get_tensor_by_name("softmax_linear/softmax_output:0")
            print(out_softmax)
            input_x = sess.graph.get_tensor_by_name("input:0")
            print(input_x)

            image_batch, label_batch = get_images_and_label(test_image_path, 16, '.png')
            image_open = tf.read_file(tf.cast(image_batch, tf.string))
            image = tf.image.decode_png(image_open, channels=3)
            for i in image:
                plt.imshow(i)

            # # img_out_softmax = sess.run(out_softmax,
            # #                            feed_dict={input_x: np.reshape(image_array, [16, fullSize, fullSize, 3])})
            # img_out_softmax = sess.run(out_softmax,
            #                            feed_dict={input_x: np.reshape(image_array)})
            # print("img_out_softmax:", img_out_softmax)
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print("label:", prediction_labels)
            #
            # # print('true label:', mnist.test.labels[0])


if __name__ == '__main__':
    print('read:' + get_lastest_pbfile(logs_dir) + '\n')
    pb_file = get_lastest_pbfile(logs_dir)
    eval_single_picture(pb_file, test_image_path)
