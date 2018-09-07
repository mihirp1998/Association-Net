
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import cv2
from keras import optimizers

from dataset import BatchGenerator, get_data
from AssociationNetModel import *
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard

data = get_data()
gen = BatchGenerator(data)
data_size = 5


left_input = Input(shape= [255, 255, 3])
right_input = Input(shape= [255, 255, 3])


with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2

global_step = tf.Variable(0, trainable=False)

left_out = mynewnet(left_input)
right_out = mynewnet(right_input)


# left side of the network
leftModel = Model(left_input,left_out)

# right side of the network
rightModel = Model(right_input,right_out)


inp1= Input(shape =[8, 8, 384])
inp2= Input(shape =[8, 8, 384])

yVal = newJoin(inp1,inp2)

# concatenating bot the sides
classifier1 = Model([inp1,inp2],yVal)


final_left = leftModel.output
final_right = rightModel.output
final_output = classifier1([final_left,final_right])

# entire model
seq = Model([leftModel.input,rightModel.input],final_output)

# get data
b_l, b_r, b_sim = gen.next_batch(data_size)

sgd = optimizers.SGD()

seq.compile(loss="binary_crossentropy",optimizer = sgd)

tensorboard =TensorBoard(log_dir='./logs')

seq.fit([b_l,b_r],b_sim,batch_size=1,epochs=10,callbacks=[tensorboard])

# save weights
leftModel.save_weights("weights.h5")