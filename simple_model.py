"""
[TEST] Simple model with datasets :
    MNIST

    one FC Layer
"""
from __future__ import print_function

import tensorflow as tf
import data_processing

def weight_variable(shape, name=None) :
    init = tf.truncated_normal_initializer(stddev=0.1)
    if name == None:
        name = "Weight"
    else :
        name = name + " Weight"

    weight = tf.get_variable(name, dtype=tf.float32, 
                                shape=shape, initializer=init)
    return weight

def bias_variable(shape, name=None) :
    init = tf.constant(0, shape=shape, dtype=tf.float32)
    if name == None:
        name = "Bias"
    else :
        name = name + " Bias"

    bias = tf.get_variable(name, dtype=tf.float32, 
                                shape=shape, initializer=init)
    return bias 


def FCLayer(input_, output_size, act="relu", name="FC-Layer") :
    input_dim = input_.get_shape()[1]
    w_shape = [input_dim, output_size]
    b_shape = [output_size]

    weight = weight_variable(w_shape)
    bias = bias_variable(b_shape)

    layer = tf.add(tf.matmul(input_, weight), bias)

    if act == "relu":
        layer = tf.nn.relu(layer)
    else :
        #[XXX] Processing other activation function 
        pass

    return layer

def Neural_Network(x) :
    layer1 = FCLayer(x, hidden1)
    
    return layer1


def model_fn(x_input, y_label, mode) :
    

