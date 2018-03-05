from .LSTM_old import *

GLOVE_DIM = 300
QUESTION_LEN = 30
LSTM_DIM = 600

def lstmFeatureExtractor(model, dataFile = '../snli_1.0/trainData_300d_840B.pkl'):
	data = pickle.load(open(dataFile, 'rb'))
	dataSize = len(data)
	feats = np.zeros((dataSize, LSTM_DIM))
	labels = np.zeros((dataSize, 3))
	for i in range(dataSize):
		sen1, sen2 = getLSTMInputs(data, i)
		feats[i, :] = model.evaluate([sen1,sen2])
		if data[i]['gold_label'] == 'entailment':
			labels[i][0] = 1
		elif data[i]['gold_label'] == 'contradiction':
			labels[i][1] = 1
		else:
			labels[i][2] = 1
	np.save('../snli_1.0/lstm_feats_train.npy', feats)
	np.save('../snli_1.0/labels_train.npy', labels)

def getLSTMInputs(data, i):
	x1 = np.zeros((1, QUESTION_LEN, GLOVE_DIM))
	x2 = np.zeros((1, QUESTION_LEN, GLOVE_DIM))

    if len(data[i]['sentence1']) < QUESTION_LEN:
        x1[0, 0: len(data[i]['sentence1'])] = data[i]['sentence1']
    else:
        x1[0, :] = data[i]['sentence1'][0:self.questionLen]

    if len(data[i]['sentence1']) < QUESTION_LEN:
        x2[0, 0: len(data[i]['sentence2'])] = data[i]['sentence2']
    else:
        x2[0, :] = data[i]['sentence2'][0:self.questionLen]
    return x1, x2


def getPartModel(weightsFile):
	sen1 = Input(shape=(self.questionLen,self.gloveDim))
	sen2 = Input(shape=(self.questionLen,self.gloveDim))
	LSTM_shared=LSTM(self.outLen, activation='tanh')

	encoded1 = LSTM_shared(sen1)
	encoded2 = LSTM_shared(sen2)
	merged = keras.layers.concatenate([encoded1,encoded2])
	

	x=Dropout(self.dropoutRate)(merged)

	reg=keras.regularizers.l2(self.regularize)

	# 3 hidden layers
	x=Dense(2*self.outLen,activation=self.activation)(x)
	x = BatchNormalization()(x)
	#x=Dropout(self.dropoutRate)(x)
	x=Dense(2*self.outLen,activation=self.activation)(x)
	x = BatchNormalization()(x)
	#x=Dropout(self.dropoutRate)(x)
	x=Dense(2*self.outLen,activation=self.activation)(x)
	x = BatchNormalization()(x)
	#x=Dropout(self.dropoutRate)(x)

	# classifier
	predictions=Dense(3,activation='softmax')(x)

	modelFull = Model(inputs=[sen1, sen2], outputs=predictions)
	modelFull.load_weights(weightsFile)
	modelPart = Model(inputs=[sen1, sen2], outputs=merged)
	print("Model created")
	return modelPart


