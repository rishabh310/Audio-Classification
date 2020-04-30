import module
import predict
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

def cnn():

    features_df = pd.read_json("my.json")

    #convert into numpy array
    X, y, le = module.get_numpy_array(features_df)

    # split into training and testing data
    X_train, X_test, y_train, y_test = module.get_train_test(X,y)
    num_labels = y.shape[1]
        
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
        
    # create model architecture
    model = create_cnn(num_labels)

    # train model
    print("Training..")
    module.train(model,X_train, X_test, y_train, y_test,"trained_cnn.h5")

    # compute test loss and accuracy
    test_loss, test_accuracy = module.compute(X_test,y_test,"trained_cnn.h5")
    print("Test loss",test_loss)
    print("Test accuracy",test_accuracy)

    # predicting using trained model with any test file in dataset
    predict.predict("dataset/001 - Dog bark/1-30226-A.ogg","trained_cnn.h5")

def create_cnn(num_labels):

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":

    cnn()