import csv

import numpy as np
import pandas as pd


class InputGenerator:
	def __init__(self, path, batch_size, train=True, mode='basic'):
		self.path = path
		self.train = train
		self.mode = mode
		self.batch_size = batch_size
		self.epoch_size = -1
		self.batch_len = -1
		self.inputs_size = -1
		self.inputs_col_num = -1
		self.batch_flag = 0
		self.num_classes = 5
		self.pad_inputs_num = -1
		
		origin_data = pd.read_csv(self.path)
		
		if train:
			origin_data.sample(frac=1)  # shuffle
			self.batch_len = origin_data.shape[0] // self.batch_size
			self.inputs_size = self.batch_len * self.batch_size
		else:
			self.inputs_size = origin_data.shape[0]
			remain_inputs_num = origin_data.shape[0] % self.batch_size
			self.pad_inputs_num = self.batch_size - remain_inputs_num
			self.inputs_size = self.inputs_size + self.pad_inputs_num
			self.batch_len = self.inputs_size // self.batch_size
		self.pet_ID = origin_data.get('PetID').tolist()
		if train:
			self.labels = origin_data.get('AdoptionSpeed').to_numpy(dtype=float)
			self.input_labels = self.labels_generator()
		else:
			self.input_labels = np.zeros([self.inputs_size + self.pad_inputs_num, 5])
		
		# self.rescuerID_voc = self._create_rescuerID_dict(origin_data.get('RescuerID'))
		
		if mode == 'basic':
			self.inputs = self.input_basic_generator(origin_data)
		else:
			self.inputs = self.input_desc_generator(origin_data)
	
	def _create_name_dict(self, name_data):
		pass
	
	def _create_desc_dict(self, desc_data):
		pass
	
	def _create_rescuerID_dict(self, rescuerID_data):
		rescuerID_dict = {}
		for i in range(rescuerID_data.size):
			rescuerID_dict[rescuerID_data.get(i)] = float(i)
		
		with open('../data/rescuerID_voc.csv', 'w') as csv_file:
			writer = csv.writer(csv_file)
			for key, value in rescuerID_dict.items():
				writer.writerow([key, value])
		
		return rescuerID_dict
	
	def read_rescuerID_voc(self):
		with open('../data/rescuerID_voc.csv') as csv_file:
			reader = csv.reader(csv_file)
			rescuerID_dict = dict(reader)
		return rescuerID_dict
	
	def input_basic_generator(self, data):
		data = data.drop(['Name', 'Description', 'State', 'RescuerID',
		                  'PetID', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3'], axis=1)
		if self.train:
			data = data.drop('AdoptionSpeed', axis=1)
		# Normalize some data
		# change cat type from 2 to -1
		# change No to -1, Not Sure to 0
		data['Type'] = data['Type'].map({1: 1, 2: -1})
		data['Gender'] = data['Gender'].map({1: 1, 2: -1, 3: 0})
		data['Vaccinated'] = data['Vaccinated'].map({1: 1, 2: -1, 3: 0})
		data['Dewormed'] = data['Dewormed'].map({1: 1, 2: -1, 3: 0})
		data['Sterilized'] = data['Sterilized'].map({1: 1, 2: -1, 3: 0})
		
		input_basic = data.to_numpy(dtype=float)
		if self.train:
			inputs = input_basic[0:self.batch_len * self.batch_size, :]
		else:
			pad = np.zeros((self.pad_inputs_num, input_basic.shape[1]))
			inputs = np.concatenate((input_basic, pad), axis=0)
			self.pet_ID = self.pet_ID + [0 for x in range(self.pad_inputs_num)]
		
		self.inputs_col_num = np.shape(inputs)[1]
		# Time major
		inputs = np.transpose(inputs)
		return inputs
	
	def labels_generator(self):
		input_labels = np.zeros((self.inputs_size, self.num_classes))
		for x in range(self.inputs_size):
			input_labels[x][int(self.labels[x])] = float(1)
		return input_labels
	
	def input_desc_generator(self, data):
		pass
	
	# def input_basic_produce(self):
	
	# return inputs
	
	def next_batch(self):
		
		if self.mode == 'basic':
			if self.inputs_size > self.batch_flag:
				next_inputs = self.inputs[:, self.batch_flag:self.batch_flag + self.batch_size, np.newaxis]
				next_labels = self.input_labels[self.batch_flag:self.batch_flag + self.batch_size, :]
				next_PetID = self.pet_ID[self.batch_flag:self.batch_flag + self.batch_size]
				self.batch_flag = self.batch_flag + self.batch_size
				return next_inputs, next_labels, next_PetID
			elif self.inputs_size == self.batch_flag:
				raise StopIteration
			else:
				raise ValueError('Iterator does not match the data size!')
		else:
			pass

# gen = InputGenerator('../data',20)
# k = gen.next_batch()
# pass
