

import pandas as pd
import numpy as np
df=pd.read_csv('criminal_train.csv')
#df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop(['Criminal','PERID'],axis=1))
scaled_features = scaler.transform(df.drop(['Criminal','PERID'],axis=1))
X = pd.DataFrame(scaled_features,columns=df.columns[1:-1])
y=df.iloc[:,-1]


X_train=X.iloc[0:30000,:]
X_val=X.iloc[30000:,:]
y_train=y.iloc[0:30000]
y_val=y.iloc[30000:]

"""
from keras.utils import np_utils
y_train= np_utils.to_categorical(y,num_classes=2)

kernel_initializer='random_uniform',
"""

"""
count =0
for i in df['Criminal']:
    if i ==0:
        count=count+1
"""




from keras.models import Sequential
from keras.layers import Dense ,Dropout,LSTM

model = Sequential()

model.add(Dense(70, input_dim=70,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


"""
from sklearn.utils.class_weight import compute_class_weight
class_weight_list = compute_class_weight('balanced', np.unique(y), y)
class_weight = dict(zip(np.unique(y), class_weight_list))
"""


"""
import keras.backend as K
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
"""

import keras

from keras import metrics

dir(keras.metrics)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist=model.fit(X_train, y_train, epochs=40, batch_size=20,validation_data=(X_val,y_val))

hist_dic=hist.history

tra_acc=hist_dic['acc']
val_acc=hist_dic['val_acc']

import matplotlib.pyplot as plt
epochs=range(1,len(tra_acc)+1)
plt.plot(epochs,tra_acc,'bo',label='training_accu')
plt.plot(epochs,val_acc,'b',label='val_accu')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()

np.set_printoptions(suppress=True)
df2=pd.read_csv('criminal_test.csv')
scaler.fit(df2.drop(['100100'],axis=1))
scaled_features1 = scaler.transform(df2.drop(['100100'],axis=1))
X_test = pd.DataFrame(scaled_features1,columns=df2.columns[1:71])
pre=model.predict(X_test)
y_test=df2.iloc[:,-1]

evalu=model.evaluate(X_test,y_test)


evalu[0]

"""
ma=np.max(pre)
ma
pre1=pre/ma
round(pre1[0])
pre2=np.round(pre1)
len(pre2)

"""
"""
pred=[]
for i in range(11430):
    if pre[i][0]>pre[i][1]:
        pred.append(0)
    else:
        pred.append(1)
    
"""
"""
pre2=[]
for i in range(11430):
    if (pre[i]<0.8):
        pre2.append(int(0))
    else:
        pre2.append(int(1) )

#np.savetxt('prediction.csv',pre2,fmt='%0.0f')

df2 = df2.rename(columns={'100100': 'PERID'})
dataframe1=df2
dataframe1['Criminal']=pre2
db=dataframe1.iloc[:,[0,71]]
db.to_csv('test_Predictions.csv',index=None)

"""


