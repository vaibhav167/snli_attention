import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from keras import regularizers


class AttentionModel:
	def __init__(self, embedDim = 100, sentenceLen = 30):
		self.embedDim = embedDim
		self.sentenceLen = sentenceLen

	def getModel(self):
		premise = Input(shape=(self.sentenceLen, self.embedDim))

