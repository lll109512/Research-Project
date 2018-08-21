import loadData as ld
import tensorflow as tf
import numpy as np
import datetime
from scipy import spatial

batch_size = ld.batch_size
maxLength = ld.maxLength



#Models:

def OneLayerCNN(name, word_vector_embedding, word_vector_embedding_dim, l2_constrain, filter_region, output_embedding_dim, feature_map_size=30):
    with tf.variable_scope(name):
        inputs = tf.placeholder(
            shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
        embedding = tf.nn.embedding_lookup(word_vector_embedding, inputs, name=f"{name}_embedding")
        shaped_batch = tf.reshape(embedding, [-1, maxLength, word_vector_embedding_dim, 1], name=f"{name}_shaped_batch")
        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.2), tf.random_normal_initializer(0., 0.1)
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_constrain)
        convs = []
        for region in filter_region:
            conv = tf.layers.conv2d(shaped_batch, feature_map_size, [region, word_vector_embedding_dim], strides=(1, 1), padding="valid", activation=tf.nn.relu,
                kernel_initializer=w_initializer, bias_initializer=b_initializer, kernel_regularizer=l2_regularizer, name=f"{region}_filter")
            pooling = tf.layers.max_pooling2d(
                conv, [conv.shape[1], conv.shape[2]], 1)
            conv_flatten = tf.layers.flatten(pooling)
            convs.append(conv_flatten)
        concated = tf.concat(convs, 1)
        fc = tf.layers.batch_normalization(tf.layers.dense(concated, output_embedding_dim,
             kernel_initializer=w_initializer, bias_initializer=b_initializer, kernel_regularizer=l2_regularizer))
    return inputs, fc


def BiLSTMmax(name, word_vector_embedding, word_vector_embedding_dim, output_embedding_dim):
    assert output_embedding_dim % 2 == 0
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")
    forward_cell = tf.contrib.rnn.LSTMCell(
        output_embedding_dim / 2, name='forward_cell', reuse=tf.AUTO_REUSE)
    backward_cell = tf.contrib.rnn.LSTMCell(
        output_embedding_dim / 2, name='backward_cell', reuse=tf.AUTO_REUSE)
    outputs , _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embedding,dtype=tf.float32)
    concatedOutput = tf.concat(outputs, 2)
    pooling = tf.layers.max_pooling1d(concatedOutput, maxLength, 1, padding='valid')
    vector = tf.reshape(pooling, [-1, output_embedding_dim])
    return inputs, vector


def BiLSTMmaxWithMultichannel(name, word_vector_embedding, word_vector_embedding_dim, output_embedding_dim):
    assert output_embedding_dim % 2 == 0
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    with tf.variable_scope("second_channel_embedding", reuse=tf.AUTO_REUSE):
        trainable_embedding = tf.get_variable(
            'trainable_embedding', shape=[word_vector_embedding.shape[0], word_vector_embedding.shape[1]], 
            initializer=tf.constant_initializer(word_vector_embedding, dtype=tf.float32, verify_shape=True))
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")
    second_channel_embedding = tf.nn.embedding_lookup(
        trainable_embedding, inputs, name=f"{name}_embedding")
    concated_embedding = tf.concat([embedding,second_channel_embedding],2)
    forward_cell = tf.contrib.rnn.LSTMCell(
        output_embedding_dim / 2, name='forward_cell', reuse=tf.AUTO_REUSE)
    backward_cell = tf.contrib.rnn.LSTMCell(
        output_embedding_dim / 2, name='backward_cell', reuse=tf.AUTO_REUSE)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell, backward_cell, concated_embedding, dtype=tf.float32)
    concatedOutput = tf.concat(outputs, 2)
    pooling = tf.layers.max_pooling1d(
        concatedOutput, maxLength, 1, padding='valid')
    vector = tf.reshape(pooling, [-1, output_embedding_dim])
    return inputs, vector


def BiMultiLSTMmaxWithMultichannel(name, word_vector_embedding, word_vector_embedding_dim, output_embedding_dim):
    def get_cell(var_name, num_units):
        return tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(num_units, name=f"first_{var_name}", reuse=tf.AUTO_REUSE),
             tf.contrib.rnn.LSTMCell(num_units, name=f"output_{var_name}", reuse=tf.AUTO_REUSE)]
        )
    assert output_embedding_dim % 2 == 0
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    with tf.variable_scope("second_channel_embedding", reuse=tf.AUTO_REUSE):
        trainable_embedding = tf.get_variable(
            'trainable_embedding', shape=[word_vector_embedding.shape[0], word_vector_embedding.shape[1]], 
            initializer=tf.constant_initializer(word_vector_embedding, dtype=tf.float32, verify_shape=True))
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")
    second_channel_embedding = tf.nn.embedding_lookup(
        trainable_embedding, inputs, name=f"{name}_embedding")
    concated_embedding = tf.concat([embedding, second_channel_embedding], 2)
    forward_cell = get_cell('forward_cell', output_embedding_dim / 2)
    backward_cell = get_cell('backward_cell', output_embedding_dim / 2)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell, backward_cell, concated_embedding, dtype=tf.float32)
    concatedOutput = tf.concat(outputs, 2)
    pooling = tf.layers.max_pooling1d(
        concatedOutput, maxLength, 1, padding='valid')
    vector = tf.reshape(pooling, [-1, output_embedding_dim])
    return inputs, vector


def BiMultiLSTMmax(name, word_vector_embedding, word_vector_embedding_dim, output_embedding_dim):
    def get_cell(var_name,num_units):
        return tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(num_units, name=f"first_{var_name}", reuse=tf.AUTO_REUSE),
             tf.contrib.rnn.LSTMCell(num_units, name=f"output_{var_name}", reuse=tf.AUTO_REUSE)]
        )

    assert output_embedding_dim % 2 == 0
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")

    forward_cell = get_cell('forward_cell',output_embedding_dim / 2)
    backward_cell = get_cell('backward_cell',output_embedding_dim / 2)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell, backward_cell, embedding, dtype=tf.float32)
    concatedOutput = tf.concat(outputs, 2)
    pooling = tf.layers.max_pooling1d(
        concatedOutput, maxLength, 1, padding='valid')
    vector = tf.reshape(pooling, [-1, output_embedding_dim])
    return inputs, vector


def OneLayerCNNWithBiLSTMMax(name, word_vector_embedding, word_vector_embedding_dim, filter_region, output_embedding_dim, feature_map_size=30):
    assert output_embedding_dim % 2 == 0
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")
    shaped_batch = tf.reshape(
        embedding, [-1, maxLength, word_vector_embedding_dim, 1], name=f"{name}_shaped_batch")
    w_initializer, b_initializer = tf.random_normal_initializer(
        0., 0.2), tf.random_normal_initializer(0., 0.1)
    convs = []
    for region in filter_region:
        if region%2 == 1:
            pad = tf.pad(shaped_batch, [[0, 0], [region//2,region//2], [0, 0], [0, 0]])
        else:
            pad = tf.pad(shaped_batch, [[0, 0], [region//2-1, region//2], [0, 0], [0, 0]])
        conv = tf.layers.conv2d(pad, feature_map_size, [region, word_vector_embedding_dim], strides=(1, 1), padding="valid", activation=tf.nn.relu,
                                kernel_initializer=w_initializer, bias_initializer=b_initializer, name=f"{region}_filter", reuse=tf.AUTO_REUSE)
        convs.append(conv)
    concated = tf.concat(convs, 3)
    cnnSeries = tf.reshape(concated, [-1, maxLength,feature_map_size * len(filter_region)])

    forward_cell = tf.contrib.rnn.LSTMCell(output_embedding_dim / 2, name='forward_cell', reuse=tf.AUTO_REUSE)
    backward_cell = tf.contrib.rnn.LSTMCell(output_embedding_dim / 2, name='backward_cell', reuse=tf.AUTO_REUSE)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        forward_cell, backward_cell, cnnSeries, dtype=tf.float32)
    concatedOutput = tf.concat(outputs, 2)
    pooling = tf.layers.max_pooling1d(concatedOutput, maxLength, 1, padding='valid',name='1_max_pooling')
    vector = tf.reshape(pooling, [-1, output_embedding_dim])
    return inputs, vector

#A model I have tried but not desribed in report.
def DCNN(name, word_vector_embedding, word_vector_embedding_dim, max_conv_layers, k_top, output_embedding_dim, feature_map_size=30, l2_constrain=3.0):
    def fold_k_max_pooling(x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                # [batch_size, k1, num_filters[1]]
                try:
                    fold = tf.add(input_unstack[i], input_unstack[i + 1])
                except IndexError:
                    fold = input_unstack[i]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                # [batch_size, num_filters[1], top_k]
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            # [batch_size, k2, embed_size/2, num_filters[1]]
            fold = tf.stack(out, axis=2)
        return fold

    
    inputs = tf.placeholder(
        shape=[None, maxLength], name=f"{name}_input", dtype=tf.int32)
    embedding = tf.nn.embedding_lookup(
        word_vector_embedding, inputs, name=f"{name}_embedding")
    conv = tf.reshape(
        embedding, [-1, maxLength, word_vector_embedding_dim, 1], name=f"{name}_shaped_batch")
    w_initializer, b_initializer = tf.random_normal_initializer(
        0., 0.2), tf.random_normal_initializer(0., 0.1)
    for current_layer in range(1,max_conv_layers+1):
        kl = max(k_top, np.ceil((max_conv_layers - current_layer)* maxLength / max_conv_layers))
        padded = tf.pad(conv,[[0,0],[1,1],[0,0],[0,0]])
        conv = tf.layers.conv2d(padded, feature_map_size,
            [3, 1], padding='same', activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, 
            name=f"{current_layer}_filter",reuse=tf.AUTO_REUSE)
        conv = fold_k_max_pooling(conv, kl)

    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_constrain)
    fc_input = tf.layers.flatten(conv)
    vector = tf.layers.dense(fc_input, output_embedding_dim, activation=tf.tanh, kernel_regularizer=l2_regularizer,
        kernel_initializer=w_initializer, bias_initializer=b_initializer,name='DCNN_full_connection',reuse=tf.AUTO_REUSE)

    return inputs, vector
            

