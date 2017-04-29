from keras.layers import LSTM, Input
from keras.engine import Model
import keras.backend as K
import numpy as np
import tensorflow as tf
# all numbers are arbitrarily picked
batch=10
seq=10
feat=10
rhid=10
test_x = np.random.random((batch,seq,feat))
test_y = np.random.random((batch, seq, rhid)) ## going to be computed against R, so needs to be same size
test_y *= 10.
xin = Input(batch_shape=(batch,seq,feat))
R = LSTM(rhid, return_sequences=True, stateful=True)
out = R(xin)
M = Model(input=xin, output=out)
M.compile('Adam', 'MSE')
sess = tf.Session()
#func = lambda r: K.mean(r.states[0]).eval()
#print(func(R))
R.reset_states()
print R.states
#print(func(R))
for _ in range(5):
    test_x = np.random.random((batch,seq,feat))
    test_y = np.random.random((batch, seq, rhid)) ## going to be computed against R, so needs to be same size
    M.train_on_batch(test_x, test_y)
    print R.states[1]
    #print(func(R))

R.reset_states()
#print(func(R))
for _ in range(5):
    test_x = np.random.random((batch,seq,feat))
    test_y = np.random.random((batch, seq, rhid)) ## going to be computed against R, so needs to be same size
    M.train_on_batch(test_x, test_y)
 #   print(func(R))