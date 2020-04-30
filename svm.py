# LINEAR KERNEL
import module
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def svm():

    features_df = pd.read_json("my.json")

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    # features_df.info()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    svc = SVC(kernel='linear', C=1)
    svc = svc.fit(X_train, y_train)
    acc = svc.score(X_test, y_test)
    print("Accuracy",acc*100)


if __name__ == "__main__":

    svm()