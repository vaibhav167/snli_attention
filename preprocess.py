import jsonlines
import numpy as np

class Preprocessor:
	def __init__(self, directory = '../snli_1.0/'):
		self.train_data = []

	def isSampleValid(self, sample):
		return sample['gold_label'] != '-'

	def getDataFromJSONL(self, filename):
		data = []
		numInvalid = 0
		with open(filename) as fp:
			reader = jsonlines.Reader(fp)
			for obj in reader.iter(type=dict, skip_invalid=True):
				if self.isSampleValid(obj):
					sample = {}
					sample['gold_label'] = obj['gold_label']
					sample['sentence1'] = obj['sentence1']
					sample['sentence2'] = obj['sentence2']
					data.append(sample)
				else:
					numInvalid+=1
		print len(data)
		print numInvalid
		return data

	def getEmbeddings(self)




preproc = Preprocessor()
data = preproc.getData("../snli_1.0/snli_1.0_train.jsonl")
print data[:10]
