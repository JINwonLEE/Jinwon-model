"""
    Author : Jinwon Lee
    Purpose : Utils for making neural net models
    Date : 2019 - 03 - 30

"""


from __future__ import print_function

import tensorflow as tf

def weight_variable(shape, name=None) :
    init = tf.truncated_normal_initializer(stddev=0.1)

    w_name = 'Weight' + name
    weight = tf.get_variable(w_name, dtype=tf.float32, 
                                shape=shape, initializer=init)
    return weight

def bias_variable(shape, name=None) :
    init = tf.constant(0, shape=shape, dtype=tf.float32)

    b_name = 'Bias' + name
    bias = tf.get_variable(b_name, dtype=tf.float32, 
                                 initializer=init)
    return bias 

def FCLayer(input_, output_size, layer_count, act="relu", name="Fclayer") :

    input_dim = input_.get_shape()[1]
    w_shape = [input_dim, output_size]
    b_shape = [output_size]

    l_name = name + str(layer_count)

    weight = weight_variable(w_shape, l_name)
    bias = bias_variable(b_shape, l_name)

    layer = tf.add(tf.matmul(input_, weight), bias)

    if act == "relu":
        layer = tf.nn.relu(layer)
    else :
        #[XXX] Processing other activation function 
        pass

    return layer

