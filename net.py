from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense

class Net:
    @staticmethod
    def build(width, height, depth, weightsPath=None):
        """
        Builds a CNN with 8 layers 
        width: target width of input
        height: target height of input
        depth: target depth of input
        weigthsPath: preload trained weights
        """
        model = Sequential()
    
        model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(width, height, depth)))
        model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
        model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
        model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(29, activation='softmax'))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            print('weights loaded')
            model.load_weights(weightsPath)

        return model