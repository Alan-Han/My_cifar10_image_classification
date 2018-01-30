import tensorflow as tf

def conv2d_maxpool(x_tensor, conv_outputs_num, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    w_shape = [conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_outputs_num]
    weight = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_outputs_num))
    y_conv = tf.nn.conv2d(x_tensor, weight, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    y_add_bias = tf.nn.bias_add(y_conv, bias)
    y_add_fn = tf.nn.relu(y_add_bias)
    y_pool = tf.nn.max_pool(y_add_fn, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return y_pool

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    x_shape = x_tensor.get_shape().as_list()
    x_flatten = tf.reshape(x_tensor, (-1, x_shape[1]*x_shape[2]*x_shape[3]))
    return x_flatten

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    w = tf.Variable(tf.truncated_normal([x_tensor.get_shape().as_list()[1], num_outputs], stddev=0.1))
    b = tf.Variable(tf.zeros(num_outputs))
    y = tf.matmul(x_tensor, w)
    y = tf.nn.bias_add(y, b)
    fc = tf.nn.relu(y)
    return fc

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    w = tf.Variable(tf.truncated_normal([x_tensor.get_shape().as_list()[1], num_outputs], stddev=0.1))
    b = tf.Variable(tf.zeros(num_outputs))
    y = tf.matmul(x_tensor, w)
    out = tf.nn.bias_add(y, b)
    return out


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    y1 = conv2d_maxpool(x, 32, [3, 3], [2, 2], [2, 2], [2, 2])
    y2 = conv2d_maxpool(y1, 64, [3, 3], [2, 2], [2, 2], [2, 2])

    #   flatten(x_tensor)
    y2_flatten = flatten(y2)

    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(y2_flatten, 1024)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc2 = fully_conn(fc1_drop, 256)
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    #   output(x_tensor, num_outputs)
    out = output(fc2_drop, 10)
    return out

