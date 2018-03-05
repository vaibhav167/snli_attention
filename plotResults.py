import pickle
import matplotlib.pyplot as plt
import numpy as np


history_singleLSTM = pickle.load(open("history.pkl", 'rb'))
history_separateLSTM = pickle.load(open("history.3_4_12.pkl", 'rb'))

print history_singleLSTM['acc']
print history_singleLSTM['val_acc']
print history_separateLSTM['acc']
print history_separateLSTM['val_acc']
# plt.figure()
plt.plot(np.arange(1, len(history_singleLSTM['acc']) + 1), history_singleLSTM['acc'])
plt.plot(np.arange(1, len(history_singleLSTM['val_acc']) + 1), history_singleLSTM['val_acc'])
plt.plot(np.arange(1, len(history_separateLSTM['acc']) + 1), history_separateLSTM['acc'])
plt.plot(np.arange(1, len(history_separateLSTM['val_acc']) + 1), history_separateLSTM['val_acc'])
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Model1, training set', 'Model1, validation set', 'Model2, training set', 'Model2, validation set'])
plt.grid()
plt.savefig('singleVseparate.PNG')