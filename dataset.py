import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import choice, permutation
from itertools import combinations
from cv2 import imread

flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator():

	def __init__(self, data):
		np.random.seed(0)
		random.seed(0)
		self.centered = data['a']['videos']
		self.decentered = data['a_decenter']['videos']
		self.i = 5
		self.to_img = lambda x: imread(x)

	def next_batch(self, batch_size):
		left = []
		right = []
		sim = []
		# centered = a['a']['videos']
		# decentered = a['a']['videos']
		# genuine
		for i in range(len(self.centered))*5:
			n = 5
			centered = self.centered[i]

			left.append(self.to_img( centered[ choice(range(len(centered) ) ) ]))
			right.append(self.to_img( centered[choice( range(len(centered) ) ) ] )) 

			sim.append([1])

		#impostor
		for i in range(len(self.centered))*5:
			centered = self.centered[i]
			decentered = self.decentered[i]

			left.append(self.to_img( centered[choice( range(len(centered)) ) ]))
			right.append(self.to_img( decentered[choice( range(len(decentered))  )] ))
			sim.append([0])
		return np.array(left), np.array(right), np.array(sim)
		

def get_mnist():
	import pickle
	a = pickle.load(open('data/train_imdb1.pickle','rb'))
	return a