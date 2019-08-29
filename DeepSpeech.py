#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time

from datetime import datetime
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from evaluate import evaluate
from six.moves import zip, range
from tensorflow.python.tools import freeze_graph, strip_unused_lib
from util.config import Config, initialize_globals
from util.feeding import create_dataset, samples_to_mfccs, audiofile_to_features
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar


# Graph Creation
# ==============

def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device(Config.cpu_device):
        # Create or get apropos variable
        var = tfv1.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x



def dense(name, x, units, dropout_rate=None, relu=True):
    with tfv1.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tfv1.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tfv1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

    if dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output


def rnn_impl_cudnn_rnn(x, seq_length, previous_state):
    assert previous_state is None # 'Passing previous state not supported with CuDNN backend'

    # Hack: CudnnLSTM works similarly to Keras layers in that when you instantiate
    # the object it creates the variables, and then you just call it several times
    # to enable variable re-use. Because all of our code is structure in an old
    # school TensorFlow structure where you can just call tf.get_variable again with
    # reuse=True to reuse variables, we can't easily make use of the object oriented
    # way CudnnLSTM is implemented, so we save a singleton instance in the function,
    # emulating a static function variable.
    if not rnn_impl_cudnn_rnn.cell:
        # Forward direction cell:
        fw_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                 num_units=Config.n_cell_dim,
                                                 input_mode='linear_input',
                                                 direction='unidirectional',
                                                 dtype=tf.float32)
        rnn_impl_cudnn_rnn.cell = fw_cell

    output, output_state = rnn_impl_cudnn_rnn.cell(inputs=x,
                                                   sequence_lengths=seq_length)

    return output, output_state

rnn_impl_cudnn_rnn.cell = None


def create_model(batch_x, seq_length, dropout, reuse=False, batch_size=None, previous_state=None, overlap=True):
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(input=batch_x)[0]

    # Create overlapping feature windows if needed
    if overlap:
        batch_x = create_overlapping_windows(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(a=batch_x, perm=[1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0])
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1])
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2])

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_size, Config.n_hidden_3])

    # Run through parametrized RNN implementation, as we use different RNNs
    # for training and inference
    output, output_state = rnn_impl_cudnn_rnn(layer_3, seq_length, previous_state)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation
    layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5])

    # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
    layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers


# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(iterator, dropout, reuse):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_filenames, (batch_x, batch_seq_len), batch_y = iterator.get_next()

    # Calculate the logits of the batch
    logits, _ = create_model(batch_x, batch_seq_len, dropout, reuse=reuse)

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Check if any files lead to non finite loss
    non_finite_files = tf.gather(batch_filenames, tfv1.where(~tf.math.is_finite(total_loss)))

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(input_tensor=total_loss)

    # Finally we return the average loss
    return avg_loss, non_finite_files


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer():
    optimizer = tfv1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                         beta1=FLAGS.beta1,
                                         beta2=FLAGS.beta2,
                                         epsilon=FLAGS.epsilon)
    return optimizer


# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def get_tower_results(iterator, optimizer, dropout_rates):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    # Aggregate any non finite files in the batches
    tower_non_finite_files = []

    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss, non_finite_files = calculate_mean_edit_distance_and_loss(iterator, dropout_rates, reuse=i > 0)

                    # Allow for variables to be re-used by the next tower
                    tfv1.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    tower_non_finite_files.append(non_finite_files)

    avg_loss_across_towers = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    tfv1.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    all_non_finite_files = tf.concat(tower_non_finite_files, axis=0)

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, all_non_finite_files


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(Config.cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads


def create_computation_inference_graph():
    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = samples_to_mfccs(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs') 

def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None

    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = samples_to_mfccs(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs')

    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    # This shape is read by the native_client in DS_CreateModel to know the
    # value of n_steps, n_context and n_input. Make sure you update the code
    # there if this shape is changed.
    input_tensor = tfv1.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
    seq_length = tfv1.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    logits, layers = create_model(batch_x=input_tensor,
                                  seq_length=seq_length if not FLAGS.export_tflite else None,
                                  dropout=no_dropout,
                                  batch_size=batch_size,
                                  previous_state=previous_state,
                                  overlap=False)
    
    # Apply softmax for CTC decoder
    logits = tf.nn.softmax(logits, name='logits')

    if batch_size <= 0:
        if n_steps > 0:
            raise NotImplementedError('dynamic batch_size expect n_steps to be dynamic too')
        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
            },
            layers
        )

    new_state_c, new_state_h = layers['rnn_output_state']
    new_state_c = tf.identity(new_state_c, name='new_state_c')
    new_state_h = tf.identity(new_state_h, name='new_state_h')

    inputs = {
        'input': input_tensor,
        'previous_state_c': previous_state_c,
        'previous_state_h': previous_state_h,
        'input_samples': input_samples,
    }

    inputs['input_lengths'] = seq_length

    outputs = {
        'outputs': logits,
        'new_state_c': new_state_c,
        'new_state_h': new_state_h,
        'mfccs': mfccs,
    }

    return inputs, outputs, layers

def file_relative_read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

from tensorflow.python import pywrap_tensorflow

def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    from tensorflow.python.framework.ops import Tensor, Operation

    inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)
    output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor)]
    output_names_ops = [op.name for op in outputs.values() if isinstance(op, Operation)]
    output_names = ",".join(output_names_tensors + output_names_ops)

    # Create a saver using variables from the above newly created graph
    def fixup(name):
        if name.startswith('cudnn_compatible_lstm_cell/'):
            return name.replace('cudnn_compatible_lstm_cell/', 'lstm_fused_cell/')
        return name
    map2 = {v.op.name: v for v in tfv1.global_variables()}
    print(map2)
    mapping = {fixup(v.op.name): v for v in tfv1.global_variables()}
    print(mapping)
    saver = tfv1.train.Saver(mapping)
    # saver = tfv1.train.Saver()





    # Restore variables from training checkpoint
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # print(checkpoint)
    checkpoint_path = checkpoint.model_checkpoint_path


    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(var_to_shape_map)



    output_filename = 'output_graph.pb'
    if FLAGS.remove_export:
        if os.path.isdir(FLAGS.export_dir):
            log_info('Removing old export')
            shutil.rmtree(FLAGS.export_dir)
    try:
        output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

        if not os.path.isdir(FLAGS.export_dir):
            os.makedirs(FLAGS.export_dir)

        def do_graph_freeze(output_file=None, output_node_names=None, variables_blacklist=''):
            frozen = freeze_graph.freeze_graph_with_def_protos(
                input_graph_def=tfv1.get_default_graph().as_graph_def(),
                input_saver_def=saver.as_saver_def(),
                input_checkpoint=checkpoint_path,
                output_node_names=output_node_names,
                restore_op_name=None,
                filename_tensor_name=None,
                output_graph=output_file,
                clear_devices=False,
                variable_names_blacklist=variables_blacklist,
                initializer_nodes='')

            input_node_names = []
            return strip_unused_lib.strip_unused(
                input_graph_def=frozen,
                input_node_names=input_node_names,
                output_node_names=output_node_names.split(','),
                placeholder_type_enum=tf.float32.as_datatype_enum)

        frozen_graph = do_graph_freeze(output_node_names=output_names)
        frozen_graph.version = int(file_relative_read('GRAPH_VERSION').strip())
            
        # Add a no-op node to the graph with metadata information to be loaded by the native client
        metadata = frozen_graph.node.add()
        metadata.name = 'model_metadata'
        metadata.op = 'NoOp'
        metadata.attr['sample_rate'].i = FLAGS.audio_sample_rate
        metadata.attr['feature_win_len'].i = FLAGS.feature_win_len
        metadata.attr['feature_win_step'].i = FLAGS.feature_win_step
        if FLAGS.export_language:
            metadata.attr['language'].s = FLAGS.export_language.encode('ascii')
            
        with open(output_graph_path, 'wb') as fout:
            fout.write(frozen_graph.SerializeToString())
        
        log_info('Models exported at %s' % (FLAGS.export_dir))
    except RuntimeError as e:
        log_error(str(e))



import tensorrt as trt

def exportTensorRTEngine():
    #convert-to-uff frozen_inference_graph.pb
    #model_file = '/data/mnist/mnist.uff'
    model_path = FLAGS.uff_file

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # if trt_engine

    # inputs = {
    #     'input': input_tensor,
    #     'previous_state_c': previous_state_c,
    #     'previous_state_h': previous_state_h,
    #     'input_samples': input_samples,
    # }
    # outputs = {
    #     'outputs': logits,
    #     'new_state_c': new_state_c,
    #     'new_state_h': new_state_h,
    #     'mfccs': mfccs,
    # }


    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        max_batch_size = 3
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = 1 <<  20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.

        # parser.register_input("previous_state_c")#Placeholder
        # parser.register_input("previous_state_h")#Placeholder
        # parser.register_input("input_samples")#Placeholder

        parser.register_output("logits")#softmax
        parser.register_output("new_state_c")#Identity
        parser.register_output("new_state_h")#identity
        parser.register_output("mfccs")#identity
        parser.parse(model_path, network)

        with builder.build_cuda_engine(network) as engine:
            with open("output_graph_trt.engine", "wb") as f:
                    f.write(engine.serialize())            
    # Do inference here.

import tensorflow.python.ops as ops

def writeTensorBoard():
    def get_graph_def_from_file(graph_filepath):
        tf.reset_default_graph()
        with ops.Graph().as_default():
            with tf.gfile.GFile(graph_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                return graph_def

    filename="/hd/mf/deepspeech-models/deepspeech-0.6.0-models-split-cudnn/output_graph.pb"
    graph_def =get_graph_def_from_file(filename)
    for node in graph_def.node:
        if node.op=='Placeholder':
            print(node) # this will be the input node
    
    with tf.Session(graph=tf.Graph()) as session:
        mygraph = tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(logdir='log_tb/1', graph=session.graph)
        writer.flush()

def main(_):
    initialize_globals()

    if FLAGS.export_dir:
        tfv1.reset_default_graph()
        export()

    if FLAGS.tensorboard:
        writeTensorBoard()

    if FLAGS.export_tensorrt_engine:
        exportTensorRTEngine()

if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
