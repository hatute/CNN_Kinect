import tensorflow as tf
import Storage as st

# TRAIN_MODE, PARAMS
MODEL_FLAGS = dict()
MODEL_FLAGS_TRAIN_MODE = 'TRAIN_MODE'
MODEL_FLAGS_TENSOR = 'TENSOR'


def is_train_mode():
    return MODEL_FLAGS[MODEL_FLAGS_TRAIN_MODE]


def get_variable(name, shape=None):
    tensor_dic = MODEL_FLAGS[MODEL_FLAGS_TENSOR]
    if is_train_mode():
        init = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
        var = tf.get_variable(name, shape, dtype=tf.float32, initializer=init)
        tensor_dic[name] = var
        return var
    else:
        return tensor_dic[name]


def build_model(input_ph, batch_size, image_channels, sample_kind):
    # --------------------------------
    # Conv 1
    #       [filter_height, filter_width, in_channels, out_channels]
    kernel_1 = get_variable('Conv1-Kernel', [1, 1, image_channels, 8])
    #       Must have `strides[0] = strides[3] = 1`
    #       [1, stride, stride, 1]
    conv_1 = tf.nn.conv2d(input_ph, kernel_1, [1, 1, 1, 1], padding='SAME')
    bias_1 = get_variable('Conv1-Bias', [8])
    pre_act_1 = tf.nn.bias_add(conv_1, bias_1)
    conv_rlt_1 = tf.nn.relu(pre_act_1)
    # Pool & Norm 1
    #       'NHWC'
    pool_1 = tf.nn.max_pool(conv_rlt_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # --------------------------------
    # Conv 2
    kernel_2 = get_variable('Conv2-Kernel', [3, 3, 8, 32])
    conv_2 = tf.nn.conv2d(norm_1, kernel_2, [1, 1, 1, 1], padding='SAME')
    bias_2 = get_variable('Conv2-Bias', [32])
    pre_act_2 = tf.nn.bias_add(conv_2, bias_2)
    conv_rlt_2 = tf.nn.relu(pre_act_2)
    # Norm & Pool 2
    norm_2 = tf.nn.lrn(conv_rlt_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # --------------------------------
    # Local 3
    # Convert from 'NWHC' to 'NX': X->1 dim data stream
    reshape = tf.reshape(pool_2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight_3 = get_variable('Local3-Weight', [dim, 128])
    bias_3 = get_variable('Local3-Bias', [128])
    local_3 = tf.nn.relu(tf.matmul(reshape, weight_3) + bias_3)
    # --------------------------------
    # Local 4
    weight_4 = get_variable('Local4-Weight', [128, 64])
    bias_4 = get_variable('Local4-Bias', [64])
    local_4 = tf.nn.relu(tf.matmul(local_3, weight_4) + bias_4)
    # --------------------------------
    # linear layer(WX + b),
    # TODO: Remove the comments
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    # TODO: KIND = 3
    weight_fin = get_variable('Final-Widght', [64, sample_kind])
    bias_fin = get_variable('Final-Bias', [sample_kind])
    final = tf.add(tf.matmul(local_4, weight_fin), bias_fin)

    return final, conv_rlt_1, conv_rlt_2


def train_operation(batch_size, image_width, image_height, image_channel, sample_kind):
    MODEL_FLAGS[MODEL_FLAGS_TRAIN_MODE] = True
    MODEL_FLAGS[MODEL_FLAGS_TENSOR] = dict()

    input_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_width, image_height, image_channel])
    label_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, sample_kind])
    model, c1, c2 = build_model(input_ph, batch_size, image_channel, sample_kind)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=label_ph, logits=model)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # TODO: Adjust learning rate
    opt = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(cross_entropy_mean, global_step=global_step)

    return train_op, input_ph, label_ph, model, c1, c2


def evaluate_opration(path, image_width, image_height, image_channel, sample_kind):
    MODEL_FLAGS[MODEL_FLAGS_TRAIN_MODE] = False
    MODEL_FLAGS[MODEL_FLAGS_TENSOR] = dict()
    load_params(path)

    input_ph = tf.placeholder(dtype=tf.float32, shape=[1, image_width, image_height, image_channel])
    model, c1, c2 = build_model(input_ph, 1, image_channel, sample_kind)
    # evaluate_op = tf.argmax(model)

    return model, input_ph, c1, c2


def save_params(path, sess):
    dic = dict()
    tensor_dic = MODEL_FLAGS[MODEL_FLAGS_TENSOR]
    for key in tensor_dic.keys():
        val = tensor_dic[key].eval(sess)
        dic[key] = val
    st.write_params(path, dic)


def load_params(path):
    dic = st.read_params(path)
    tensor_dic = MODEL_FLAGS[MODEL_FLAGS_TENSOR]
    for key in dic.keys():
        var = tf.Variable(dic[key], name=key)
        tensor_dic[key] = var
