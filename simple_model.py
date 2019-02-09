"""
[TEST] Simple model with datasets :
    MNIST

    one FC Layer
"""
from __future__ import print_function

import tensorflow as tf
import data_processing


learning_rate = 0.01
batch_size = 128
number_of_iteration = 1000

hidden1 = 256
out_class = 10


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

def Neural_Network(x) :
    layer1 = FCLayer(x, hidden1, 0)
    layer2 = FCLayer(layer1, out_class, 1)
    
    return layer2


def model_fn(features, labels, mode) :
    logit = Neural_Network(features)

    predict_class = tf.argmax(logit, axis=1)
    #predict_probability = tf.nn.softmax(logit)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predict_class)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss_op,
                                  global_step=global_step)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict_class)

# Create hook list which is for logging some losses, accuracy in certain step 
    train_hook_list = []
    train_tensor_log = {'accuracy_1' : accuracy[1], 'loss' : loss_op, 'global_step' : global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensor_log, every_n_iter=100))

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predict_class,
        loss=loss_op,
        train_op=train_op,
        training_hooks=train_hook_list,
        eval_metric_ops={'accuracy': accuracy})

    return estim_specs

# Visualize the process of training steps
tf.logging.set_verbosity(tf.logging.INFO)

# Try getting the dataset if there is no data set in local 
data_processing.try_download_and_extract()

# This is the case of MNIST
# In case of other data set (Cifar10, Cifar100) you have to check
mnist = data_processing.get_dataset()

config = None
# Logging Device placement 
#config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=True))

# Create Estimator with model directory [XXX] Model Directory check
model = tf.estimator.Estimator(model_fn, model_dir="/Users/jwl1993/model_dir", config=config)

# Training input, if num_epochs == None, training will run forever until the max_step is set
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=mnist.train.images, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

_ = model.train(input_fn, steps=number_of_iteration)


# Test input
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=mnist.test.images, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)

evaluation = model.evaluate(input_fn)

print("Testing Accuracy:", evaluation['accuracy'])
