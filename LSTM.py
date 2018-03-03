from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import pickle
import numpy as np
import keras
from sklearn.cross_validation import train_test_split
import os
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


class lstmModel:
    def __init__(self,questionLen=30,gloveDim=300,outLen=300,dropoutRate=0.2,regularize=0.01,activation='tanh'):
        self.questionLen=questionLen
        self.gloveDim=gloveDim
        self.outLen=outLen
        self.dropoutRate=dropoutRate
        self.regularize=regularize
        self.activation=activation
        self.getModelSingleLSTM()
        self.trainCounter = 0
        self.valCounter = 0

    def getTrainGenerator(self):
    	"""
    	returns a generator for train data
    	"""
    	while 1:
    		yield self.getTrainBatch(batchSize = 32)

    def getValGenerator(self):
    	while 1:
    		yield self.getValBatch(batchSize = 32)

    def getModel(self):
        # LSTM layer transforms the two input sentences
        sen1 = Input(shape=(self.questionLen,self.gloveDim))
        sen2 = Input(shape=(self.questionLen,self.gloveDim))
        shared_LSTM=LSTM(self.outLen)

        encoded1 = shared_LSTM(sen1)
        encoded2 = shared_LSTM(sen2)
        merged = keras.layers.concatenate([encoded1,encoded2])
        x = Dropout(self.dropoutRate)(merged)

        reg = keras.regularizers.l2(self.regularize)

        # 3 hidden layers
        x = Dense(2*self.outLen,activation=self.activation)(x)
        #x=Dropout(self.dropoutRate)(merged)
        x = Dense(2*self.outLen,activation=self.activation)(x)
        #x=Dropout(self.dropoutRate)(merged)
        x = Dense(2*self.outLen,activation=self.activation)(x)
        #x=Dropout(self.dropoutRate)(merged)

        # classifier
        predictions = Dense(3,activation='softmax')(x)

        self.model = Model(inputs=[sen1, sen2], outputs=predictions)
        print("Model created")
        return self.model

    def getModelSingleLSTM(self):
        # LSTM layer transforms the two input sentences
        X = Input(shape=(2*self.questionLen + 1, self.gloveDim))
        x = BatchNormalization()(X)
        lstm1 = LSTM(self.outLen, activation = self.activation, return_sequences = True, dropout = 0.2, name = 'lstm1')
        lstm2 = LSTM(self.outLen, activation = self.activation, return_sequences = False, dropout = 0.2, name = 'lstm2')

        x = lstm1(x)
        x = BatchNormalization()(x)
        x = lstm2(x)
        x = BatchNormalization()(x)
        #x=Dropout(self.dropoutRate)(merged)
        reg = keras.regularizers.l2(self.regularize)
        # 3 hidden layers
        x = Dense(2*self.outLen,activation=self.activation)(x)
        x = BatchNormalization()(x)
        # x=Dropout(self.dropoutRate)(x)
        x = Dense(2*self.outLen,activation=self.activation)(x)
        x = BatchNormalization()(x)
        # x=Dropout(self.dropoutRate)(x)
        x = Dense(2*self.outLen,activation=self.activation)(x)
        x = BatchNormalization()(x)
        # x=Dropout(self.dropoutRate)(x)
        

        # classifier
        predictions=Dense(3,activation='softmax')(x)

        self.model = Model(inputs = [X], outputs=predictions)
        print("Model created")
        return self.model

    def compile(self):
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()


    def train(self,numEpochs,batchSize):
        # checkpoint
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        #self.model.load_weights(filepath)
        self.model.fit(self.trainX, self.trainY, epochs = numEpochs, batch_size = batchSize, validation_data = (self.testX, self.testY), callbacks=callbacks_list)
        print("Training complete")

    def trainDistr(self,numEpochs,batchSize):
        # checkpoint
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        #self.model.load_weights(filepath)
        self.model.fit_generator(self.getTrainGenerator(), steps_per_epoch = self.trainSize//batchSize, epochs = numEpochs, 
        	validation_data = self.getValGenerator(), validation_steps = (self.totalSize - self.trainSize)//batchSize, callbacks=callbacks_list)
        print("Training complete")


    def trainSingleLSTM(self,numEpochs,batchSize):
		trainHist = [[], []]
		valHist = [[], []]
		iterSize = 100
		bestLoss = 100
		weightFilePath = "./1234"
		nIter = self.trainSize/(batchSize)
		for i in range(numEpochs):
		    for j in range(nIter):
		        trainX, trainY = self.getTrainBatch(batchSize = batchSize)
		        self.model.fit( trainX, trainY, verbose = False )
		        # print stats for every 'iterSize' iterations
		        if( j%iterSize == 0 ):
		            trainX, trainY = self.getTrainBatch(batchSize = 5*batchSize)
		            valX, valY = self.getValBatch(batchSize = 5*batchSize)
		            trainLoss, trainAcc = self.model.evaluate( trainX, trainY, verbose = False )
		            valLoss, valAcc = self.model.evaluate(valX, valY, verbose = False )
		            if valLoss < bestLoss:
		                bestLoss = valLoss
		                print("Best loss:{0:.3f}\t BessAcc: {1:.3f} \t Saving weights ".format(valLoss, valAcc))
		                if os.path.isfile(weightFilePath):
		                    os.remove(weightFilePath)
		                weightFilePath = 'model/weights_loss_{0:.4f}_acc_{1:.3f}.h5py'.format(valLoss, valAcc)
		                self.model.save_weights(weightFilePath)
		            print( "Epoch:{0}\t Iter: {1}\t TrainLoss {2:.4f}\t ValidLos: {3:.4f} \
		            TrainAcc: {4:.4f}\t ValidAcc: {5:.4f}"
		                   .format(i, j+1, trainLoss, valLoss, trainAcc, valAcc))
		            trainHist[0].append(trainLoss)
		            trainHist[1].append(trainAcc)
		            valHist[0].append(valLoss)
		            valHist[1].append(valAcc)
		np.save('trainHistory.npy', np.array(trainHist))
		np.save('valHistory.npy', np.array(valHist))

    def importData(self,dataFile,train=True):
		self.data=pickle.load(open(dataFile, 'rb'))
		print("data unpickled")

		#initialize with zeros for padding
		self.trainX=np.zeros((len(self.data),2*self.questionLen+1,self.gloveDim))

		if train==True:
		    self.labels=np.zeros((len(self.data)))

		#pick first 30 words of each sentence, assign integers to labels
		for i in range(0,len(self.data)):
		    if len(self.data[i]['sentence1']) < self.questionLen:
		        self.trainX[i,0:len(self.data[i]['sentence1'])] = self.data[i]['sentence1']
		    else:
		        self.trainX[i,0:self.questionLen]=self.data[i]['sentence1'][0:self.questionLen]

		    fillPtr = min(self.questionLen, len(self.data[i]['sentence1']))

		    # leave a zero vector for end token of premise
		    fillPtr+=1


		    if len(self.data[i]['sentence2']) < self.questionLen:
		        self.trainX[i,fillPtr:fillPtr + len(self.data[i]['sentence2'])] = self.data[i]['sentence2']
		    else:
		        self.trainX[i,fillPtr:fillPtr + self.questionLen] = self.data[i]['sentence2'][0:self.questionLen]

		    if train==True:
		        if self.data[i]['gold_label']=='entailment':
		            self.labels[i]=1;

		        elif self.data[i]['gold_label']=='contradiction':
		            self.labels[i]=2;

		        elif self.data[i]['gold_label']!='neutral':
		            print("Error: problem with gold label")

		# free some memory
		self.data=None

		if train==True:
		    self.trainY=keras.utils.to_categorical(self.labels, num_classes=3)
		    #shuffle data and label in unison

		    self.testX, self.trainX = self.trainX[:100000,:,:], self.trainX[100000:,:,:]
		    self.testY, self.trainY = self.trainY[:100000,:], self.trainY[100000:,:]
		    # self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.trainX, self.one_hot, test_size = 0.2)
		

		print("data ready for training")


    def getTrainBatch(self, batchSize = 32):
		if self.trainCounter >= self.trainSize - 32 :
			self.trainCounter = 0

		trainX = np.zeros((batchSize,2*self.questionLen+1,self.gloveDim))
		trainY = np.zeros((batchSize))

		for i in range(0,batchSize):
			if len(self.data[self.trainCounter + i]['sentence1']) < self.questionLen:
				trainX[i,0:len(self.data[self.trainCounter + i]['sentence1'])] = self.data[self.trainCounter + i]['sentence1']
			else:
				trainX[i,0:self.questionLen] = self.data[self.trainCounter + i]['sentence1'][0:self.questionLen]

			fillPtr = min(self.questionLen, len(self.data[i]['sentence1']))

			# leave a zero vector for end token of premise
			fillPtr+=1


			if len(self.data[self.trainCounter + i]['sentence2']) < self.questionLen:
				trainX[i,fillPtr:fillPtr + len(self.data[self.trainCounter + i]['sentence2'])] = self.data[self.trainCounter + i]['sentence2']
			else:
				trainX[i,fillPtr:fillPtr + self.questionLen] = self.data[self.trainCounter + i]['sentence2'][0:self.questionLen]

			
			if self.data[self.trainCounter + i]['gold_label']=='entailment':
				trainY[i]=1;
			elif self.data[self.trainCounter + i]['gold_label']=='contradiction':
				trainY[i]=2;
			elif self.data[self.trainCounter + i]['gold_label']!='neutral':
				print("Error: problem with gold label")

		trainY = keras.utils.to_categorical(trainY, num_classes=3)

		self.trainCounter += 32
		return trainX, trainY

    def getValBatch(self, batchSize = 32):
		if self.valCounter >= self.totalSize - 32 :
		    self.valCounter = self.trainSize

		trainX = np.zeros((batchSize,2*self.questionLen+1,self.gloveDim))
		trainY = np.zeros((batchSize))

		for i in range(0,batchSize):
			if len(self.data[self.valCounter + i]['sentence1']) < self.questionLen:
				trainX[i,0:len(self.data[self.valCounter + i]['sentence1'])] = self.data[self.valCounter + i]['sentence1']
			else:
				trainX[i,0:self.questionLen] = self.data[self.valCounter + i]['sentence1'][0:self.questionLen]

			fillPtr = min(self.questionLen, len(self.data[i]['sentence1']))

			# leave a zero vector for end token of premise
			fillPtr+=1


			if len(self.data[self.valCounter + i]['sentence2']) < self.questionLen:
				trainX[i,fillPtr:fillPtr + len(self.data[self.valCounter + i]['sentence2'])] = self.data[self.valCounter + i]['sentence2']
			else:
				trainX[i,fillPtr:fillPtr + self.questionLen] = self.data[self.valCounter + i]['sentence2'][0:self.questionLen]

			
			if self.data[self.valCounter + i]['gold_label']=='entailment':
				trainY[i]=1;
			elif self.data[self.valCounter + i]['gold_label']=='contradiction':
				trainY[i]=2;
			elif self.data[self.valCounter + i]['gold_label']!='neutral':
				print("Error: problem with gold label")
		trainY = keras.utils.to_categorical(trainY, num_classes=3)
		self.valCounter += 32
		return trainX, trainY

    def importDataDistr(self, dataFile):
    	self.data=pickle.load(open(dataFile, 'rb'))
    	self.totalSize = len(self.data)
    	self.trainSize = 440000
    	self.valCounter = self.trainSize

	def getTrainBatchDistr(self, batchSize = 32):
		if self.trainCounter >= self.trainSize - 32 :
			self.trainCounter = 0

		trainX = np.zeros((batchSize,2*self.questionLen+1,self.gloveDim))
		trainY = np.zeros((batchSize))

		for i in range(0,batchSize):
			if len(self.data[i]['sentence1']) < self.questionLen:
				trainX[i,0:len(self.data[self.trainCounter + i]['sentence1'])] = self.data[self.trainCounter + i]['sentence1']
			else:
				trainX[i,0:self.questionLen] = self.data[self.trainCounter + i]['sentence1'][0:self.questionLen]

			fillPtr = min(self.questionLen, len(self.data[i]['sentence1']))

			# leave a zero vector for end token of premise
			fillPtr+=1


			if len(self.data[i]['sentence2']) < self.questionLen:
				trainX[i,fillPtr:fillPtr + len(self.data[self.trainCounter + i]['sentence2'])] = self.data[self.trainCounter + i]['sentence2']
			else:
				trainX[i,fillPtr:fillPtr + self.questionLen] = self.data[self.trainCounter + i]['sentence2'][0:self.questionLen]

			
			if self.data[i]['gold_label']=='entailment':
				trainY[i]=1;
			elif self.data[i]['gold_label']=='contradiction':
				trainY[i]=2;
			elif self.data[i]['gold_label']!='neutral':
				print("Error: problem with gold label")

		trainY = keras.utils.to_categorical(trainY, num_classes=3)

		self.trainCounter += 32
		return trainX, trainY


	def getValBatchDistr(self, batchSize = 32):
		if self.valCounter >= self.totalSize - 32 :
		    self.valCounter = self.trainSize

		trainX = np.zeros((batchSize,2*self.questionLen+1,self.gloveDim))
		trainY = np.zeros((batchSize))

		for i in range(0,batchSize):
			if len(self.data[i]['sentence1']) < self.questionLen:
				trainX[i,0:len(self.data[self.valCounter + i]['sentence1'])] = self.data[self.valCounter + i]['sentence1']
			else:
				trainX[i,0:self.questionLen] = self.data[self.valCounter + i]['sentence1'][0:self.questionLen]

			fillPtr = min(self.questionLen, len(self.data[i]['sentence1']))

			# leave a zero vector for end token of premise
			fillPtr+=1


			if len(self.data[i]['sentence2']) < self.questionLen:
				trainX[i,fillPtr:fillPtr + len(self.data[self.valCounter + i]['sentence2'])] = self.data[self.valCounter + i]['sentence2']
			else:
				trainX[i,fillPtr:fillPtr + self.questionLen] = self.data[self.valCounter + i]['sentence2'][0:self.questionLen]

			
			if self.data[i]['gold_label']=='entailment':
				trainY[i]=1;
			elif self.data[i]['gold_label']=='contradiction':
				trainY[i]=2;
			elif self.data[i]['gold_label']!='neutral':
				print("Error: problem with gold label")

		trainY = keras.utils.to_categorical(trainY, num_classes=3)
	    
		self.valCounter += 32
		return trainX, trainY



    # def trainGenerator(self, batchSize = 32):
    # 	while(1):
    # 		yield self.getTrainBatch(batchSize = batchSize)

    # def validGenerator(self, batchSize = 32):
    # 	while(1):
    #         yield self.getValidBatch(batchSize = batchSize)

mymodel=lstmModel()
mymodel.compile()
mymodel.importDataDistr('../snli_1.0/trainData_300d_42B.pkl')
mymodel.trainDistr(15,32)
