
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import cv2
from keras import optimizers

from dataset import BatchGenerator, get_mnist
from model import *
from keras.models import Model,Sequential
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 5000, 'Total training iter')
flags.DEFINE_integer('step', 500, 'Save after ... iteration')

mnist = get_mnist()
gen = BatchGenerator(mnist)
b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)


print('shape is ',b_l.shape,b_r.shape)

# have a look at the dataset
# for i in range(b_l.shape[0]):
# 	cv2.imshow('bl ',b_l[i])
# 	cv2.waitKey(0)
# 	cv2.imshow('br ',b_r[i])

# 	cv2.waitKey(0)
# 	print(b_sim[igg])
# 	cv2.destroyAllWindows()



# test_im = np.array([im.reshape((28,28,1)) for im in mnist.test.images])
c = ['#ff0000', '#ffff00']


left = tf.placeholder(tf.float32, [None, 255, 255, 3], name='left')
right = tf.placeholder(tf.float32, [None, 255, 255, 3], name='right')
left_input = Input(shape= [255, 255, 3])
right_input = Input(shape= [255, 255, 3])


with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2

# left_output = mynet(left, reuse=False)
# print("left +",left_output)

# right_output = mynet(right, reuse=True)

# loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)

# train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

left_out = mynewnet(left_input)
right_out = mynewnet(right_input)
# joined =  newJoin(left_out,right_out)

print("lefting ",left_out)

leftModel = Model(left_input,left_out)
rightModel = Model(right_input,right_out)


inp1= Input(shape =[8, 8, 384])
inp2= Input(shape =[8, 8, 384])

yVal = newJoin(inp1,inp2)

classifier1 = Model([inp1,inp2],yVal)
print("output",leftModel.output)

final_left = leftModel.output
final_right = rightModel.output
final_output = classifier1([final_left,final_right])

seq = Model([leftModel.input,rightModel.input],final_output)
# print(seq.summary())


b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

sgd = optimizers.SGD()

seq.compile(loss="binary_crossentropy",optimizer = sgd)

seq.fit([b_l,b_r],b_sim,batch_size=1,epochs=10)

# print("amodel weights layer1  ",aModel.layers[1].get_weights)


# aModel.fit([b_l,b_r],b_sim,batch_size=1,epochs=10)
leftModel.save_weights("model.h5")

# saver = tf.train.Saver()

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())

# 	# setup tensorboard	
# 	tf.summary.scalar('step', global_step)
# 	tf.summary.scalar('loss', loss)
	
# 	for var in tf.trainable_variables():
# 		tf.summary.histogram(var.op.name, var)

# 	merged = tf.summary.merge_all()
# 	writer = tf.summary.FileWriter('train.log', sess.graph)

# 	#train iter

# 	for i in range(FLAGS.train_iter):
# 		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)
# 		# l = sess.run(left_output ,feed_dict={left:np.random.randn(2, 128, 128, 1)})
# 		# _, l, out = sess.run([train_step, loss,left_output], feed_dict={left:b_l, right:b_r, label: b_sim})
# 		aModel.fit(b_l)
# 		print("loss is ",l)
# 		print("out shape ",out.shape)
# 		# writer.add_summary(summary_str, i)
# 		print "\r#%d - Loss"%i, l

		
		# if (i + 1) % FLAGS.step == 0:
		# 	#generate test
		# 	feat = sess.run(left_output, feed_dict={left:test_im})
			
		# 	labels = mnist.test.labels
		# 	# plot result
		# 	f = plt.figure(figsize=(16,9))
		# 	for j in range(10):
		# 	    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
		# 	    	'.', c=c[j],alpha=0.8)
		# 	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
		# 	plt.savefig('img/%d.jpg' % (i + 1))

	# saver.save(sess, "model/model.ckpt")