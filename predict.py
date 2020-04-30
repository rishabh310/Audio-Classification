import module
import numpy as np
import pandas as pd
import sys
from keras.models import load_model

def predict(filename,model_file):

    features_df = pd.read_json("my.json")
    X, y, le = module.get_numpy_array(features_df)

    model = load_model(model_file)
    prediction_feature = module.get_features(filename)
    if model_file == "trained_mlp.h5":
        prediction_feature = np.array([prediction_feature])
    elif model_file == "trained_cnn.h5":    
        prediction_feature = np.expand_dims(np.array([prediction_feature]),axis=2)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    name = features_df.Directory[predicted_class[0] * 40]
    oname =''.join([i for i in name if not i.isdigit()])
    print("Predicted class",predicted_class[0],oname)
    # print(oname)
    predicted_proba_vector = model.predict_proba([prediction_feature])

    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )



if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])



