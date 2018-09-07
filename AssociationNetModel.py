import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed, BatchNormalization,Flatten,Dense,Concatenate,Subtract

from keras.activations import relu
flags = tf.app.flags




def mynewnet(input, reuse=False):

	net = Convolution2D( 96, [7, 7],strides=2, activation=relu, padding='same',data_format="channels_last")(input)
	net = MaxPooling2D([3, 3],strides=2, padding='same',data_format="channels_last")(net)
	net = BatchNormalization()(net)

	net = Convolution2D( 384, [5, 5],strides=2, activation=relu, padding='same',data_format="channels_last")(net)
	net = MaxPooling2D([2, 2],strides=2, padding='same',data_format="channels_last")(net)
	net = BatchNormalization()(net)

	net = Convolution2D( 512, [3, 3],strides=1, activation=relu, padding='same',data_format="channels_last")(net)

	net = Convolution2D( 512, [3, 3],strides=1, activation=relu, padding='same',data_format="channels_last")(net)	

	net = Convolution2D (384, [3, 3],strides=1, activation=relu, padding='same',data_format="channels_last")(net)
	net = MaxPooling2D( [3, 3],strides=2,padding='same',data_format="channels_last")(net)

	return net 



def newJoin(leftOut,rightOut):
	netLeft = Flatten()(leftOut)
	leftFirst = Dense(4096,activation=relu)(netLeft)
	leftSecond = Dense(4096,activation=relu)(netLeft)
	leftConcat = Concatenate(axis =1)([leftFirst,leftSecond])

	netRight = Flatten()(rightOut)
	rightFirst = Dense(4096,activation=relu)(netRight)
	rightSecond = Dense(4096,activation=relu)(netRight)
	rightConcat = Concatenate(axis =1)([rightFirst,rightSecond])


	sub = Subtract()([leftConcat,rightConcat])
	one_val = Dense(1)(sub)
	return one_val	






def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		diff = model1 - model2
		one_val  = tf.layers.dense(diff,1)
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=one_val,labels=y))