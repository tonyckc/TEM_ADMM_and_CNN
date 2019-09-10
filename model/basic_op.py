# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:14:01 2019

@author: Kecheng Chen
"""
import tensorflow as tf


def conv_op(x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu,
            ):

    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)
        return z


def res_block_layers_v1(x, name, n_out_list, change_dimension=False, block_stride=1):
    if change_dimension:
        conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1, kw=1, dh=block_stride,
                dw=block_stride,
                padding="SAME", activation=None)
    else:
        short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_localConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_localConv2", n_out_list[0], training=True, useBN=True, kh=3, kw=3,
                           dh=1, dw=1,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_3 = conv_op(block_conv_2, name + "_localConv3", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                           dh=1, dw=1,
                           padding="SAME", activation=None)

    block_res = tf.add(short_cut_conv, block_conv_3)
    res = tf.nn.relu(block_res)
    return res

def res_block_layers_v2(x, name, n_out, block_stride=1):

    short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_localConv1", n_out, training=True, useBN=True, kh=3, kw=3,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_localConv2", n_out, training=True, useBN=True, kh=3, kw=3,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=None)
    block_res = tf.add(short_cut_conv, block_conv_2)
    res = tf.nn.relu(block_res)
    return res

def dilated_conv_op(x, name, n_out, training, useBN, kh=3, kw=3, stride_height=1, stride_width=1, rate_height=1,
                    rate_width=1, padding="SAME", activation=tf.nn.relu,
                    ):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=
        tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(scope + 'b', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
        conv = tf.nn.dilation2d(x, w, strides=[1, stride_height, stride_width, 1],
                                rates=[1, rate_height, rate_width, 1],
                                padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)

        return z
