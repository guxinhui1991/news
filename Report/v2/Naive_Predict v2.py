from __future__ import division
import pandas as pd
import numpy as np

msft = pd.read_csv('data\MSFT_full.csv',usecols=[0,1])
msft.sort_index(inplace=True)

raw_train = msft[:int(0.7*len(msft))]
raw_test = msft[int(0.9*len(msft)):]

msft_returns = msft['close']/msft['close'].shift(-1)

length = 5
threshold = 0.03

label = np.zeros(len(msft_returns)-length+1)
for i in range(len(label)):
    if max(np.cumprod(msft_returns[i:i+length])) > 1+threshold:
        label[i] = 1
    elif min(np.cumprod(msft_returns[i:i+length])) < 1-threshold:
        label[i] = -1
        
label_train = label[:int(0.7*len(label))]
label_test = label[int(0.9*len(label)):]



p = sum(label_train[label_train==1])/float(len(label_train))
q = -sum(label_train[label_train==-1])/float(len(label_train))

flip_coin = np.random.uniform(size=len(label_test))
up_move_predict = np.array(flip_coin > 1-p).astype(float)
down_move_predict = np.array(flip_coin < q).astype(float)
movement_predict = up_move_predict - down_move_predict

correct_predict = movement_predict == label_test

print sum(correct_predict)/float(len(label_test))



