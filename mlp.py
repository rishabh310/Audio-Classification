import module
import predict
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

def mlp():

    features_df = pd.read_json("my.json")

    #convert into numpy array
    X, y, le = module.get_numpy_array(features_df)

    # split into training and testing data
    X_train, X_test, y_train, y_test = module.get_train_test(X,y)
    num_labels = y.shape[1]

    # create model architecture
    model = create_mlp(num_labels)

    # train model
    print("Training..")
    module.train(model,X_train, X_test, y_train, y_test,"trained_mlp.h5")

    # compute test loss and accuracy
    test_loss, test_accuracy = module.compute(X_test,y_test,"trained_mlp.h5")
    print("Test loss",test_loss)
    print("Test accuracy",test_accuracy)

    # predicting using trained model with any test file in dataset
    predict.predict("sample_wav/new_file_baby.wav","trained_mlp.h5")

def create_mlp(num_labels):

    model = Sequential()
    model.add(Dense(256,input_shape = (40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256,input_shape = (40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":

    mlp()