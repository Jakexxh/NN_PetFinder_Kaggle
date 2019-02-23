import csv

import numpy as np
import pandas as pd


class InputGenerator:
	def __init__(self, path, batch_size, train=True):
		self.path = path
		self.train = train
		self.batch_size = batch_size
		self.epoch_size = -1
		self.batch_len = -1
		
		origin_data = pd.read_csv(self.path + '/train.csv')
		self.labels = origin_data.get('AdoptionSpeed').to_numpy(dtype=float)
		self.pet_ID = origin_data.get('PetID').tolist()
		
		self.rescuerID_voc = self._create_rescuerID_dict(origin_data.get('RescuerID'))
		self.input_basic = self.input_basic_generator(origin_data)
		self.input_desc = self.input_desc_generator(origin_data)
	
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
		# Normalize some data
		# change cat type from 2 to -1
		# change No to -1, Not Sure to 0
		
		data['Type'] = data['Type'].map({1: 1, 2: -1})
		data['Gender'] = data['Gender'].map({1: 1, 2: -1, 3: 0})
		data['Vaccinated'] = data['Vaccinated'].map({1: 1, 2: -1, 3: 0})
		data['Dewormed'] = data['Dewormed'].map({1: 1, 2: -1, 3: 0})
		data['Sterilized'] = data['Sterilized'].map({1: 1, 2: -1, 3: 0})
		
		input_basic = data.to_numpy(dtype=float)
		return input_basic
	
	def input_desc_generator(self, data):
		pass
	
	def input_basic_produce(self):
		self.batch_len = self.input_basic.shape[0] // self.batch_size
		inputs = self.input_basic[0:self.batch_len * self.batch_size, :]
		return inputs


class InputBasicIterator:
	def __init__(self, inputs, batch_size):
		self.inputs = inputs
		self.inputs_size = np.shape(inputs)[0]
		self.batch_size = batch_size
		self.flag = 0
	
	def __iter__(self):
		return self
	
	def __next__(self):
		
		if self.inputs_size > self.flag:
			next_inputs = self.inputs[self.flag:self.flag + self.batch_size, :, np.newaxis]
			self.flag = self.flag + self.batch_size
			return next_inputs
		elif self.inputs_size == self.flag:
			raise StopIteration
		else:
			raise ValueError('Iterator does not match the data!')

# gen = InputGenerator('../data',20)
# iterator = InputBasicIterator(gen.input_basic_produce(),gen.batch_size)
# print(next(iterator))
# print(next(iterator))
