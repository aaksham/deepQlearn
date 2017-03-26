from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class DeepQ:
    def build(self,width,height,depth,nA,weightsPath=None):
        model=Sequential()
        model.add(Convolution2D(16,8,8,strides=4,
                  input_shape=(depth,width,height)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32,4,4,strides=2))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dense(nA))
        model.add(Activation("softmax"))
        if weightsPath is not None:
            model.load_weights(weightsPath)
        return model