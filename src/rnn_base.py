import argparse

import numpy as np
import tensorflow as tf

from .data_processor import InputGenerator


def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		'--data_dir',
		type=str,
		default='../data/',
		help='data directory')
	parser.add_argument(
		'--save_dir',
		type=str,
		default='../checkpoint',
		help='directory to store checkpointed models')
	parser.add_argument(
		'--log_dir',
		type=str,
		default='../logs',
		help='directory to store  logs')
	parser.add_argument(
		'--rnn_size', type=int, default=128, help='size of RNN hidden state')
	parser.add_argument(
		'--num_layers',
		type=int,
		default=2,
		help='number of layers in the RNN')
	parser.add_argument(
		'--batch_size', type=int, default=50, help='minibatch size')
	parser.add_argument(
		'--save_every', type=int, default=1000, help='save frequency')
	parser.add_argument(
		'--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument(
		'--input_keep_prob',
		type=float,
		default=1.0,
		help='probability of keeping weights in the input layer')
	parser.add_argument(
		'--mode', type=str, default='basic', help='the mode of NN')
	parser.add_argument(
		'--train', type=bool, default=True, help='train or test')
	args = parser.parse_args()
	run(args)


def PetFinder_Basic_RNN(train, args, inputs):
	train = train
	input_size = tf.shape(inputs)[1]
	
	if train:
		keep_prob = args.input_keep_prob
	else:
		keep_prob = 1.0
	
	cell = tf.contrib.rnn.LayerNormBasicLSTMCell(args.rnn_size,
	                                             input_size=inputs.shape[1], dropout_keep_prob=keep_prob)
	
	initial_state = cell.zero_state(args.batch_size, dtype=tf.float64)
	inputs = tf.reshape(inputs, [args.batch_size, input_size, 1])
	outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
	                                         initial_state=initial_state,
	                                         dtype=tf.float64)
	
	return (outputs[-1], final_state)


def run(args):
	input_generator = InputGenerator(args.data_dir, args.batch_size, train=args.train)
	
	labels = input_generator.labels
	num_classes = np.shape(labels[0])
	
	inputs_holder = tf.placeholder("float", [None, input_generator.inputs_size, 1])
	labels_holder = tf.placeholder("float", [None, num_classes])
	train_holder = tf.placeholder(tf.bool)
	
	# Define weights
	
	with tf.name_scope('final_weights'):
		f_weight = tf.get_variable(
			'W', [args.rnn_size, num_classes],
			initializer=tf.orthogonal_initializer())
	tf.summary.histogram("f_weight", f_weight)
	
	with tf.name_scope('final_biases'):
		f_biases = tf.get_variable('b', [num_classes])
	tf.summary.histogram("f_biases", f_biases)
	
	outputs, final_state = PetFinder_Basic_RNN(train_holder, args, inputs_holder)
	logits = tf.matmul(outputs[-1], f_weight) + f_biases
	
	# logits = tf.nn.softmax(logits)
	
	# loss
	
	# quad_weighted_kappa = kappa(input_generator.labels)
	
	
	with tf.Session() as sess:
		pass
	## TODO: define loop to produce inputs

if __name__ == '__main__':
	main()
