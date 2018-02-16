import jsonlines
import numpy as np

class data:
	def __init__(self, directory = '.'):
		self.train_data = []

train_data = []
filename = "snli_1.0_train.jsonl"
with open(filename) as fp:
	reader = jsonlines.Reader(fp)
	line = reader.read()
print type(line)