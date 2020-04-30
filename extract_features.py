import module
import os
import librosa
import soundfile as sf
import numpy as np
import glob
import pandas as pd


def extract_features():

    # path to dataset containing 10 subdirectories of .ogg files
    sub_dirs = os.listdir('data')
    sub_dirs.sort()
    features_list = []
    for label, sub_dir in enumerate(sub_dirs):  
        for file_name in glob.glob(os.path.join('data',sub_dir,"*.ogg")):
            print("Extracting file ", file_name)
            try:
                mfccs = module.get_features(file_name)
            except Exception as e:
                print("Extraction error")
                continue
            features_list.append([mfccs,label, sub_dir])

    features_df = pd.DataFrame(features_list,columns = ['feature','class_label', 'Directory'])
    # print(type(features_df))
    # print(features_df)
    # df.to_csv('file_name.csv', index=False)
    # features_df.to_csv("my.csv", index=False)
    features_df.to_json("my.json")
    print(features_df)    
    return features_df


if __name__ == "__main__":

    extract_features()
