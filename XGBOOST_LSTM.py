import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pickle
import matplotlib.pyplot as plt

trainXFile = ''
trainYFile = ''
valXFile = ''
valYFile = ''
trainX = np.load(trainXFile)
trainY = np.load(trainYFile)
valX = np.load(valXFile)
valY = np.load(valYFile)


# train XGBoost Model
xgbModel = xgb.XGBClassifier(max_depth=14)
xgbModel.fit(train_feats, train_labels)



predsTrain = xgbModel.predict_proba(trainX)
predsVal = xgbModel.predict_proba(valX)

np.argmax(preds_val)