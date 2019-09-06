#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
import tensorrt as trt
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
from tensorflow.python import ops
from tensorflow.python.tools import freeze_graph, strip_unused_lib
from tensorflow.python import pywrap_tensorflow
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




def rnn_impl_deprecated_basic_rnn(x, seq_length, previous_state):
    # Forward direction cell:
    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=Config.n_cell_dim, dtype=tf.float32)
    rnn_impl_basic_rnn.cell = fw_cell
    
    output, output_state = rnn_impl_basic_rnn.cell(inputs=x[0,:,:],
                                                   state=previous_state)

    return output, output_state


def rnn_impl_basic_rnn(x, seq_length, previous_state):
    # Forward direction cell:
    fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=Config.n_cell_dim,
                                                dtype=tf.float32)
    rnn_impl_basic_rnn.cell = fw_cell
    
    output, output_state = rnn_impl_basic_rnn.cell(inputs=x[0,:,:],
                                                   state=previous_state)

    return output, output_state



def rnn_impl_cudnn_rnn(x, seq_length, previous_state):
    # assert previous_state is None # 'Passing previous state not supported with CuDNN backend'
    
    # hack: CudnnLSTM works similarly to Keras layers in that when you instantiate
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


def rnn_impl_cudnn_compatible_rnn(x, seq_length, previous_state):
    '''
    Mike's version of LSTM layer for cuda compatibility
    '''

    fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=Config.n_cell_dim)
                                                               
    rnn_impl_cudnn_compatible_rnn.cell = fw_cell
    
    # x = tf.zeros([1,x.shape[2]])
    # previous_state = [x,x]
    output_seqs, states = rnn_impl_cudnn_compatible_rnn.cell(x[0,:,:], previous_state)

    return output_seqs, states


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
    #rnn_impl=rnn_impl_basic_rnn
    rnn_impl=rnn_impl_deprecated_basic_rnn
    #rnn_impl=rnn_impl_cudnn_rnn
    #rnn_impl=rnn_impl_cudnn_compatible_rnn
    output, output_state = rnn_impl(layer_3, seq_length, previous_state)

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

        previous_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)
        #previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    logits, layers = create_model(batch_x=input_tensor,
                                  seq_length=seq_length,
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


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    from tensorflow.python.framework.ops import Tensor, Operation

    inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size,
                                                n_steps=FLAGS.n_steps,
                                                tflite=FLAGS.export_tflite)
    output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor)]
    output_names_ops = [op.name for op in outputs.values() if isinstance(op, Operation)]
    output_names = ",".join(output_names_tensors + output_names_ops)

    def fixup(name):
        for orgName in ['basic_lstm_cell/', 'lstm_cell/', 'cudnn_lstm/', 'cudnn_compatible_lstm_cell/']:
            if name.startswith(orgName):
                return name.replace(orgName, 'lstm_fused_cell/').replace('opaque_kernel', 'kernel')
        return name
 
    map2 = {v.op.name: v for v in tfv1.global_variables()}
    print("#### map2 ####")
    for i in map2.items():
        print(i)
    mapping = {fixup(v.op.name): v for v in tfv1.global_variables()}
    print("#### mapping ####")
    for i in mapping.items():
        print(i)
    saver = tfv1.train.Saver(mapping)
    # saver = tfv1.train.Saver()
    
    # Restore variables from training checkpoint
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # print(checkpoint)
    checkpoint_path = checkpoint.model_checkpoint_path


    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("#### var_to_shape_map ####")
    for i in var_to_shape_map.items():
        print(i)

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
        # frozen_graph.version = int(file_relative_read('GRAPH_VERSION').strip())
        # # Add a no-op node to the graph with metadata information to be loaded by the native client
        # metadata = frozen_graph.node.add()
        # metadata.name = 'model_metadata'
        # metadata.op = 'NoOp'
        # metadata.attr['sample_rate'].i = FLAGS.audio_sample_rate
        # metadata.attr['feature_win_len'].i = FLAGS.feature_win_len
        # metadata.attr['feature_win_step'].i = FLAGS.feature_win_step

        ### WORKING BLOCK ###
        # from tensorflow.contrib import tensorrt as trt
        # trt_graph = trt.create_inference_graph(
        #     input_graph_def=frozen_graph,  # frozen model
        #     outputs=['logits'],
        #     max_batch_size=512,  # specify your max batch size
        #     max_workspace_size_bytes=2 * (10 ** 9),  # specify the max workspace
        #     precision_mode="FP16")  # precision, can be "FP32" (32 floating point precision) or "FP16" .

        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,  # frozen model
            nodes_blacklist=['logits'],
            minimum_segment_size=6,
            max_batch_size=512,  # specify your max batch size
            max_workspace_size_bytes=2 * (10 ** 9),  # specify the max workspace
            precision_mode="FP16")  # precision, can be "FP32" or "FP16" or "INT8" .

        trt_graph = converter.convert()
        # write the TensorRT model to be used later for inference
        # tf.io.write_graph(trt_graph, "/home/ubuntu/", "trt_output_graph.pbtxt") 
        # exit()    
        with tf.io.gfile.GFile("/home/ubuntu/trt_output_graph.pb", 'wb') as g:
            g.write(trt_graph.SerializeToString())
            
        # with open(output_graph_path, 'wb') as fout:
        #     fout.write(frozen_graph.SerializeToString())
        
        log_info('Models exported at %s' % (FLAGS.export_dir))
    except RuntimeError as e:
        log_error(str(e))





def exportTensorRTEngine():
    #convert-to-uff frozen_inference_graph.pb
    #model_file = '/data/mnist/mnist.uff'
    model_path = FLAGS.uff_file
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        max_batch_size = 3
        builder.max_batch_size = max_batch_size
        # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
        builder.max_workspace_size = 1 <<  20
        
        parser.register_input("previous_state_c", [2048])#Placeholder
        parser.register_input("previous_state_h", [2048])#Placeholder
        parser.register_input("input_samples", [512])#Placeholder
        
        parser.register_output("logits")#softmax
        parser.register_output("new_state_c")#Identity
        parser.register_output("new_state_h")#identity
        parser.register_output("mfccs")#identity
        parser.parse(model_path, network)
        
        with builder.build_cuda_engine(network) as engine:
            with open("output_graph_trt.engine", "wb") as f:
                f.write(engine.serialize())
                # Do inference here.
    

def main(_):
    initialize_globals()

    if FLAGS.export_dir:
        tfv1.reset_default_graph()
        export()

    if FLAGS.export_tensorrt_engine:
        exportTensorRTEngine()

if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
