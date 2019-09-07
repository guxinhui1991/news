import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC




msft = pd.read_csv('data/MSFT_full.csv', index_col=0)
msft.index = pd.to_datetime(msft.index)

headlines = pd.read_csv("data/Headlines.csv", index_col=2)
headlines.index = pd.to_datetime(headlines.index)
headlines.head()

headlines_byDate = {}
NUM_DAYS = len(headlines_byDate)
Date_List= headlines.index.unique()

for date in headlines.index.unique():
    headlines_daily = headlines[headlines.index==date]
    str_date = ''
    for i in range(len(headlines_daily)):
        str_date= str_date+' '+(headlines_daily.Title[i]).upper()
    headlines_byDate[date] = str_date
    
dict_fin = pd.read_excel('Dictionary/LoughranMcDonald_MasterDictionary_2014_lite.xlsx', index_col=0)
word_list = np.array(dict_fin.index)
features = ['Negative', 'Positive', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous', 'Interesting', 'Modal']

import time
start_time = time.time()

Estimate_date ={}
daily_mat = pd.DataFrame(columns=features, index=dict_fin.index)
daily_mat[:] = 0.0
summy_mat = pd.DataFrame(columns=features, index=Date_List)
summy_mat[:] = 0.0

for date in Date_List:
    daily_mat[:] = 0.0
    str_date = headlines_byDate[date]
    for word in word_list:
        if word in str_date:
            for feature in features:
                if(dict_fin[dict_fin.index==word][feature].values>0):
                    daily_mat.iloc[daily_mat.index==word, daily_mat.columns==feature] = daily_mat.iloc[daily_mat.index==word, daily_mat.columns==feature] + 1
    summy_mat.iloc[summy_mat.index==date, :] = np.array(daily_mat.sum(axis=0))

print("--- %s seconds ---" % (time.time() - start_time))





def cal_indicators(data, length=5):
    '''
    data: a dictionary or dataframe contain open, close, high, low price and volume for a stock
    length: time horizon for calculating indicators
    ...
    return SMA, WMA, MFI, RSI indicators
    '''
    close_price = np.array(data['close'])
    
    SMA = np.zeros(len(data['close'])-length+1)
    WMA = np.zeros(len(data['close'])-length+1)
    MFI = np.zeros(len(data['close'])-length+1)
    RSI = np.zeros(len(data['close'])-length+1)
    
    sumSMA = np.cumsum(close_price)
    SMA[0] = sumSMA[0]
    SMA[1:] = (sumSMA[length:] - sumSMA[:-length])/length
    
    weight = np.arange(1,length+1)
    weight = weight.astype(float) / sum(weight)
    for i in range(len(WMA)):
        WMA[i] = sum(close_price[i:i+length]*weight)
    
    MoneyFlow = np.array(data['volume']*(data['high'] + data['low'] + data['close'])/float(3))
    MF_positive = np.ones(len(MoneyFlow)).astype(bool)
    MF_positive[1:] = (MoneyFlow[1:] > MoneyFlow[:-1]).astype(bool)
    for i in range(len(MFI)):
        MoneyFlow_window = MoneyFlow[i:i+length]
        MF_positive_window = MF_positive[i:i+length]
        MFR = sum(MoneyFlow_window[MF_positive_window])/sum(MoneyFlow_window[~MF_positive_window])
        MFI[i] = 100 - 100/(1+MFR)
    
    
    RS_positive = np.ones(len(MoneyFlow)).astype(bool)
    RS_positive[1:] = (close_price[1:] > close_price[:-1])
    for i in range(len(RSI)):
        RS_window = close_price[i:i+length]
        RS_positive_window = RS_positive[i:i+length]
        if sum(RS_positive_window) == 0:
            RS = 0
        elif sum(RS_positive_window) == length:
            RS = float('Inf')
        else:
            RS = np.mean(RS_window[RS_positive_window])/np.mean(RS_window[~RS_positive_window])
            
        RSI[i] = 100 - 100/(1+RS)
        
    return SMA, WMA, MFI, RSI


def cal_indicators_senti(data, length=5):
    '''
    data: a dictionary or dataframe contains daily sentiment for a stock
    length: time horizon for calculating indicators
    ...
    return SMA, WMA indicators
    '''
    sentiment = np.array(data)
    
    SMA = np.zeros([sentiment.shape[0]-length+1, sentiment.shape[1]])
    WMA = np.zeros([sentiment.shape[0]-length+1, sentiment.shape[1]])
    
    sumSMA = np.cumsum(sentiment, axis=0)
    SMA[0,:] = sumSMA[0,:]
    SMA[1:,:] = (sumSMA[length:,:] - sumSMA[:-length,:])/float(length)
    
    weight = np.arange(1,length+1)
    weight = weight.astype(float) / sum(weight)
    for i in range(len(WMA)):
        WMA[i,:] = np.asmatrix(weight)*sentiment[i:i+length,:]
        
    return SMA, WMA







length = 5
threshold = 0.03


msft.sort_index(inplace=True)

msft_returns = (msft['close'][1:].values/msft['close'][:-1].values)

Date_List_Trading = msft.index 
x_data = pd.DataFrame(index= Date_List_Trading, columns=summy_mat.columns)
for date in Date_List_Trading:
    x_data[x_data.index==date] = summy_mat[summy_mat.index==date]

x_data.sort_index(inplace=True)
x_data = x_data[:][:-length]



label = np.zeros(len(msft_returns)-length+1)
for i in range(len(label)):
    if max(np.cumprod(msft_returns[i:i+length])) > 1+threshold:
        label[i] = 1
    elif min(np.cumprod(msft_returns[i:i+length])) < 1-threshold:
        label[i] = -1

y_data = pd.DataFrame(index= msft.index[:-length], columns=['Returns'])
y_data['Returns'] = label


nan_index = (x_data['Positive'][x_data['Positive'].isnull().values]).index
x_data = x_data.drop(nan_index)
y_data = y_data.drop(nan_index)


extra_feature = {}
extra_feature['price_SMA'], extra_feature['price_WMA'], extra_feature['price_MFI'], extra_feature['price_RSI'] = cal_indicators(msft[:][:-length].drop(nan_index), length=length)
extra_feature['sentiment_SMA'], extra_feature['sentiment_WMA'] = cal_indicators_senti(x_data, length=length)

x_data = x_data[:][length-1:]
y_data = y_data[:][length-1:]

for feature in ['price_MFI', 'price_RSI']:
    x_data[feature] = extra_feature[feature]

x_data[features] = extra_feature['sentiment_WMA']


X_train = x_data[:int(0.7*len(y_data))]
X_validate = x_data[int(0.7*len(y_data)):int(0.9*len(y_data))]
X_test = x_data[int(0.9*len(y_data)):]
y_train = y_data[:int(0.7*len(y_data))]
y_validate = y_data[int(0.7*len(y_data)):int(0.9*len(y_data))]
y_test = y_data[int(0.9*len(y_data)):]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

# Model training
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, reg in enumerate([0.01, 0.1, 0.2, 0.5, 1, 1.5, 3, 5, 20, 50]):
    clf = SVC(C = reg, decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_validate)
    diff_soa = y_predict - y_validate['Returns']
    success_rate = float(len(diff_soa[diff_soa==0]))/len(diff_soa)
    print 'C = ', reg, ', success_rate = ', success_rate



clf = SVC(C = 1, decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
diff_soa = y_predict - y_test['Returns']
success_rate = float(len(diff_soa[diff_soa==0]))/len(diff_soa)
print 'C = ', reg, ', success_rate = ', success_rate
















