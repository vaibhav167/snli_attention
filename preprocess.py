import jsonlines
import numpy as np
import cPickle as pickle
import os



def getEmbedDict(gloveDir):
	embedDict = {}
	with open(os.path.join(gloveDir, 'glove.6B.100d.txt')) as fp:
		for line in fp:
			values = line.split()
			word, coef = values[0], np.asarray(values[1:], dtype='float32')
			embedDict[word] = coef
	return embedDict


def saveEmbedDictAsPickle(embedDict):
	pickle.dump(embedDict, open('../embeddings/embedDict.pkl', 'wb'))


def loadEmbedDictFromPickle(filename = '../embeddings/embedDict.pkl'):
	return pickle.load(open(filename, 'rb'))

class Preprocessor:
	def __init__(self, dataDir = '../snli_1.0/', embedDictFile = '../embeddings/embedDict.pkl', embedDim = 100):
		self.data = []
		self.embedDict = {}
		self.embedDim = embedDim
		self.embedDictFile = embedDictFile
		self.dataDir = dataDir 

	def isSampleValid(self, sample):
		return sample['gold_label'] != '-'

	def loadEmbedDict(self):
		self.embedDict = loadEmbedDictFromPickle(self.embedDictFile)

	def getDataFromJSONL(self, filename):
		numInvalid = 0
		with open(filename) as fp:
			reader = jsonlines.Reader(fp)
			for obj in reader.iter(type=dict, skip_invalid=True):
				if self.isSampleValid(obj):
					sample = {}
					sample['gold_label'] = obj['gold_label']
					sample['sentence1'] = obj['sentence1']
					sample['sentence2'] = obj['sentence2']
					self.data.append(sample)
				else:
					numInvalid+=1
		print len(self.data)
		print numInvalid
		

	def embedData(self):
		self.loadEmbedDict()
		for i in range(len(self.data)):
			sample = self.data[i]
			self.data[i] = {'gold_label': sample['gold_label'], 'sentence1': [], 'sentence2': []}
			s1 = sample['sentence1'].lower().split()
			s2 = sample['sentence2'].lower().split()
			for word in s1:
				self.data[i]['sentence1'].append(self.embedDict.get(word, np.zeros(self.embedDim)))
			for word in s2:
				self.data[i]['sentence2'].append(self.embedDict.get(word, np.zeros(self.embedDim)))
			if i%1000 == 0:
				print i, ' samples done'

	def saveDataAsPickle(self):
		pickle.dump(self.data, open(os.path.join(self.dataDir, 'trainData.pkl'), 'wb'))




		

# Save embed dict once and for all
# embedDict = getEmbedDict('../embeddings')
# saveEmbedDictAsPickle(embedDict)



# Embed data

pp = Preprocessor()
pp.getDataFromJSONL('../snli_1.0/snli_1.0_train.jsonl')
pp.embedData()
pp.saveDataAsPickle()
