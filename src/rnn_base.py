import argparse
import csv

import numpy as np
import tensorflow as tf
from data_processor import InputGenerator


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
		'--batch_size', type=int, default=30, help='minibatch size')
	parser.add_argument(
		'--save_every', type=int, default=60, help='save frequency')
	parser.add_argument(
		'--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument(
		'--input_keep_prob',
		type=float,
		default=0.8,
		help='probability of keeping weights in the input layer')
	parser.add_argument(
		'--mode', type=str, default='basic', help='the mode of NN')
	parser.add_argument(
		'--train', type=bool, default=True, help='train or test')
	args = parser.parse_args()
	run(args)


def PetFinder_Basic_RNN(train, args, inputs):
	train = train
	
	keep_prob = tf.constant(0.)
	keep_prob = tf.cond(train, lambda: tf.add(keep_prob, args.input_keep_prob), lambda: tf.add(keep_prob, 1.))
	
	cell = tf.contrib.rnn.LayerNormBasicLSTMCell(args.rnn_size, dropout_keep_prob=keep_prob)
	# cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
	# cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=keep_prob)
	# initial_state = cell.zero_state(args.batch_size, dtype=tf.float32)
	initial_state = tf.contrib.rnn.LSTMStateTuple(
		tf.truncated_normal([args.batch_size, args.rnn_size], stddev=0.1),
		tf.truncated_normal([args.batch_size, args.rnn_size], stddev=0.1))
	outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
	                                         initial_state=initial_state,
	                                         dtype=tf.float32,
	                                         time_major=True)
	
	return (outputs[-1], final_state)


def run(args):
	inputs_train_generator = InputGenerator(args.data_dir + '/train.csv', args.batch_size, train=args.train)
	inputs_test_generator = InputGenerator(args.data_dir + '/test/test.csv', args.batch_size, train=False)
	
	num_classes = inputs_train_generator.num_classes
	
	inputs_holder = tf.placeholder("float32", [inputs_train_generator.inputs_col_num, None, 1])
	labels_holder = tf.placeholder("float32", [None, num_classes])
	train_holder = tf.placeholder(tf.bool)
	
	with tf.name_scope('f_basic_weight'):
		f_weight = tf.get_variable(
			'f_basic_weight', [args.rnn_size, num_classes],
			initializer=tf.truncated_normal_initializer())
	tf.summary.histogram("f_basic_weight", f_weight)
	
	with tf.name_scope('f_basic_biases'):
		f_biases = tf.get_variable('f_basic_biases', [num_classes])
	tf.summary.histogram("f_basic_biases", f_biases)
	
	output, final_state = PetFinder_Basic_RNN(train_holder, args, inputs_holder)
	logits = tf.matmul(output, f_weight) + f_biases
	
	prediction = tf.nn.softmax(logits)
	
	# loss and optimizer
	basic_xe_loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_holder))
	tf.summary.scalar("basic_xe_loss", basic_xe_loss)
	
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_holder, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	tf.summary.scalar('basic_acc', accuracy)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
	
	gvs = optimizer.compute_gradients(basic_xe_loss)
	# capped_gvs = [(None
	#                if grad is None else tf.clip_by_value(grad, -1., 1.), var)
	#               for grad, var in gvs]
	#
	# for (grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
	# 	if grad is not None:
	# 		tf.summary.histogram('grad/{}'.format(var.name), capped_grad)
	# 		tf.summary.histogram('capped_fraction/{}'.format(var.name),
	# 		                     tf.nn.zero_fraction(grad - capped_grad))
	# 		tf.summary.histogram('variable/{}'.format(var.name), var)
	
	train_op = optimizer.apply_gradients(gvs)
	
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(args.log_dir + '/train')
	saver = tf.train.Saver()
	
	init = tf.global_variables_initializer()
	qwk_train = []
	with tf.Session() as sess:
		sess.run(init)
		for step in range(1, inputs_train_generator.batch_len + 1):
			train_inputs, train_labels, _ = inputs_train_generator.next_batch()
			sess.run(
				[train_op], feed_dict={inputs_holder: train_inputs,
				                       labels_holder: train_labels,
				                       train_holder: True})
			
			if step % args.save_every == 0 or step == 1:
				# Calculate batch loss and accuracy
				summary, loss, acc, pred, _ = sess.run(
					[merged, basic_xe_loss, accuracy, prediction, train_op],
					feed_dict={inputs_holder: train_inputs,
					           labels_holder: train_labels,
					           train_holder: True})
				
				# quad_weighted_kappa = np.mean([kappa(l, p) for l, p in zip(train_labels, pred)])
				# qwk_train.append([step, quad_weighted_kappa])
				
				print("Step " + str(step) + ", Minibatch Loss= " +
				      "{:.4f}".format(loss) + ", Training Accuracy= " +
				      "{:.4f}".format(acc))
				# + ", Training QWK= " + "{:.4f}".format(quad_weighted_kappa)
				train_writer.add_summary(summary, step)
		
		print("Optimization Finished!")
		
		preds = []
		IDs = []
		for step in range(1, inputs_test_generator.batch_len + 1):
			test_inputs, test_labels, ID = inputs_test_generator.next_batch()
			pred = sess.run([prediction],
			                feed_dict={inputs_holder: test_inputs,
			                           labels_holder: test_labels,
			                           train_holder: False})
			
			tmp_pred = [np.argmax(x) for x in pred[0]]
			preds = preds + tmp_pred
			IDs = IDs + ID
		
		with open(args.log_dir + '/submission.csv', 'w') as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(['PetID', 'AdoptionSpeed'])
			writer.writerows(zip(IDs[:- inputs_test_generator.pad_inputs_num],
			                     preds[:- inputs_test_generator.pad_inputs_num]))
		
		save_path = saver.save(sess, args.save_dir + "/model.ckpt")
		print("Model saved in path: %s" % save_path)
		train_writer.close()


if __name__ == '__main__':
	main()
