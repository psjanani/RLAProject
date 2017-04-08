from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import numpy
import os



json_file = open('/Users/janani/weights/exp_small/model0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/janani/weights/exp_small/250000_0.hd5")
print("Loaded model from disk")


for layer in loaded_model.layers:
    g=layer.get_config()
    h=layer.get_weights()
    print (g)
    if len(h) > 0:
        l = np.argmax(h[0], axis=1)
        for i in l:
            print i


# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)