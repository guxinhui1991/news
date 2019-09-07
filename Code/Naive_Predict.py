import pandas as pd
import numpy as np

raw_train = pd.read_csv('data\MSFT_train.csv',usecols=[0,1])
raw_test = pd.read_csv('data\MSFT_test.csv',usecols=[0,1])

stock_return = raw_train['close']/raw_train['close'].shift(-1)-1
stock_return_test = raw_test['close']/raw_test['close'].shift(-1)-1

up_move = stock_return[:-1] > 0
up_move_test = stock_return_test[:-1] > 0

p = up_move.mean()

flip_coin = np.random.uniform(size=len(up_move_test))
up_move_predict = flip_coin > p

correct_predict = up_move_predict == up_move_test

sum(correct_predict)



