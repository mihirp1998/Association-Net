import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed, BatchNormalization,Flatten,Dense,Concatenate,Subtract

from keras.activations import relu
flags = tf.app.flags
FLAGS = flags.FLAGS

def mynet(input, reuse=False):
	with tf.name_scope("model"):
		with tf.variable_scope("conv1") as scope:
			net = tf.contrib.layers.conv2d(input, 96, [7, 7],stride=2, activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [3, 3],stride=2)
			net = tf.contrib.layers.layer_norm(net,scope=scope,reuse=reuse)

		with tf.variable_scope("conv2") as scope:
			net = tf.contrib.layers.conv2d(net, 384, [5, 5],stride=2, activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2],stride=2, padding='SAME')
			net = tf.contrib.layers.layer_norm(net,scope=scope,reuse=reuse)

		with tf.variable_scope("conv3") as scope:
			net = tf.contrib.layers.conv2d(net, 512, [3, 3],stride=1, activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)

		with tf.variable_scope("conv4") as scope:
			net = tf.contrib.layers.conv2d(net, 512, [3, 3],stride=1, activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)	

		with tf.variable_scope("conv5") as scope:
			net = tf.contrib.layers.conv2d(net, 384, [3, 3],stride=1, activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [3, 3],stride=2)
		with tf.variable_scope("fc1") as scope:
			net = tf.contrib.layers.flatten(net)
			relu1 = tf.layers.dense(net,4096,activation=tf.nn.relu,reuse=reuse)
		with tf.variable_scope("fc2") as scope:	
			relu2 = tf.layers.dense(net,4096,activation=tf.nn.relu,reuse=reuse)	
		net = tf.concat([relu1,relu2],axis=1)


	return net




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

		# with tf.variable_scope("fc1") as scope:
		# 	net = tf.contrib.layers.flatten(net)
		# 	relu1 = tf.layers.dense(net,4096,activation=tf.nn.relu,reuse=reuse)
		# with tf.variable_scope("fc2") as scope:	
		# 	relu2 = tf.layers.dense(net,4096,activation=tf.nn.relu,reuse=reuse)	
		# net = tf.concat([relu1,relu2],axis=1)


	# return net


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