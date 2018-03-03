from keras.layers import Input, Embedding, LSTM, Dense, Dropout, TimeDistributed
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import pickle
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


class lstmModel:

    def __init__(self,questionLen=30,gloveDim=300,outLen=300,dropoutRate=0.2,regularize=0.01,activation='relu'):
        self.questionLen=questionLen
        self.gloveDim=gloveDim
        self.outLen=outLen
        self.dropoutRate=dropoutRate
        self.regularize=regularize
        self.activation=activation
        self.trainCounter = 0
        self.valCounter = 0
        self.getModel()

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

        self.model = Model(inputs=[sen1, sen2], outputs=predictions)
        print("Model created")
        return self.model

    def compile(self):
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()

    def train(self,numEpochs,batchSize):
        inputs=[self.sen1,self.sen2]

        # checkpoint
        filepath="model/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        self.model.fit(inputs,self.one_hot,epochs=numEpochs,batch_size=batchSize, validation_split = 0.2, callbacks=callbacks_list)
        print("Training complete")

    def trainDistr(self,numEpochs,batchSize):
        # checkpoint
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint, EarlyStopping(patience=4)]
        #self.model.load_weights(filepath)
        history = self.model.fit_generator(self.getTrainGenerator(), steps_per_epoch = self.trainSize//batchSize, epochs = numEpochs, 
            validation_data = self.getValGenerator(), validation_steps = (self.valSize)//batchSize, callbacks=callbacks_list)
        print("Training complete")
        pickle.dump(history, open('history.pkl', 'wb'))


    def getTrainBatch(self, batchSize = 32):
        if self.trainCounter >= self.trainSize - 32 :
            self.trainCounter = 0

        trainX1 = np.zeros((batchSize,self.questionLen,self.gloveDim))
        trainX2 = np.zeros((batchSize,self.questionLen,self.gloveDim))
        trainY = np.zeros((batchSize))

        for i in range(0,batchSize):
            if len(self.data[self.trainCounter + i]['sentence1']) < self.questionLen:
                trainX1[i,0:len(self.data[self.trainCounter + i]['sentence1'])] = self.data[self.trainCounter + i]['sentence1']
            else:
                trainX1[i,0:self.questionLen] = self.data[self.trainCounter + i]['sentence1'][0:self.questionLen]


            if len(self.data[self.trainCounter + i]['sentence2']) < self.questionLen:
                trainX2[i,:len(self.data[self.trainCounter + i]['sentence2'])] = self.data[self.trainCounter + i]['sentence2']
            else:
                trainX2[i,:self.questionLen] = self.data[self.trainCounter + i]['sentence2'][0:self.questionLen]

            
            if self.data[self.trainCounter + i]['gold_label']=='entailment':
                trainY[i]=1;
            elif self.data[self.trainCounter + i]['gold_label']=='contradiction':
                trainY[i]=2;
            elif self.data[self.trainCounter + i]['gold_label']!='neutral':
                print("Error: problem with gold label")

        trainY = keras.utils.to_categorical(trainY, num_classes=3)

        self.trainCounter += 32
        return [trainX1, trainX2], trainY

    def getValBatch(self, batchSize = 32):
        if self.valCounter >= self.valSize - 32 :
            self.valCounter = 0

        trainX1 = np.zeros((batchSize,self.questionLen,self.gloveDim))
        trainX2 = np.zeros((batchSize,self.questionLen,self.gloveDim))
        trainY = np.zeros((batchSize))

        for i in range(0,batchSize):
            if len(self.valData[self.valCounter + i]['sentence1']) < self.questionLen:
                trainX1[i,0:len(self.valData[self.valCounter + i]['sentence1'])] = self.valData[self.valCounter + i]['sentence1']
            else:
                trainX1[i,0:self.questionLen] = self.valData[self.valCounter + i]['sentence1'][0:self.questionLen]


            if len(self.valData[self.valCounter + i]['sentence2']) < self.questionLen:
                trainX2[i,: len(self.valData[self.valCounter + i]['sentence2'])] = self.valData[self.valCounter + i]['sentence2']
            else:
                trainX2[i,: self.questionLen] = self.valData[self.valCounter + i]['sentence2'][0:self.questionLen]

            
            if self.valData[self.valCounter + i]['gold_label']=='entailment':
                trainY[i]=1;
            elif self.valData[self.valCounter + i]['gold_label']=='contradiction':
                trainY[i]=2;
            elif self.valData[self.valCounter + i]['gold_label']!='neutral':
                print("Error: problem with gold label")
        trainY = keras.utils.to_categorical(trainY, num_classes=3)
        self.valCounter += 32
        return [trainX1, trainX2], trainY

    def importDataDistr(self, dataFileTrain, dataFileVal, train = True):
        self.data = pickle.load(open(dataFileTrain, 'rb'))
        self.valData = pickle.load(open(dataFileVal, 'rb'))
        self.totalSize = len(self.data)
        self.trainSize = len(self.data)
        self.valSize = len(self.valData)


    def importData(self,dataFile,train=True):
        # unpickle the data
        self.data=pickle.load(open(dataFile, 'rb'))

        print("data unpickled")

        #initialize with zeros for padding
        self.sen1=np.zeros((len(self.data),self.questionLen,self.gloveDim))
        self.sen2=np.zeros((len(self.data),self.questionLen,self.gloveDim))
        if train==True:
            self.labels=np.zeros((len(self.data)))

        #pick first 30 words of each sentence, assign integers to labels
        for i in range(0,len(self.data)):
            if len(self.data[i]['sentence1']) < self.questionLen:
                self.sen1[i,0:len(self.data[i]['sentence1'])] = self.data[i]['sentence1']
            else:
                self.sen1[i,:]=self.data[i]['sentence1'][0:self.questionLen]

            if len(self.data[i]['sentence2']) < self.questionLen:
                self.sen2[i,0:len(self.data[i]['sentence2'])] = self.data[i]['sentence2']
            else:
                self.sen2[i,:]=self.data[i]['sentence2'][0:self.questionLen]

            if train==True:
                if self.data[i]['gold_label']=='entailment':
                    self.labels[i]=1;

                elif self.data[i]['gold_label']=='contradiction':
                    self.labels[i]=2;

                elif self.data[i]['gold_label']!='neutral':
                    print("Error: problem with gold label")

        if train==True:
            self.one_hot=keras.utils.to_categorical(self.labels, num_classes=3)

# free some memory
        self.data=None

        print("data ready for training")

mymodel=lstmModel()
mymodel.importDataDistr('../snli_1.0/trainData_300d_42B.pkl', '../snli_1.0/valData_300d_42B.pkl')
mymodel.compile()
mymodel.trainDistr(25,32)