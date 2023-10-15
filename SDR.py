import warnings
import os
import joblib
from hmmlearn import hmm
import numpy as np
import librosa
from librosa.feature import mfcc
import random

def build_dataset(dir, rte):
    file_list = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    train_dataset = {}
    test_dataset = {}
    cnt = 1
    nm = int(rte * 50)
    rnd = random.sample(range(0,50), nm)
    for filename in file_list:
        label = filename.split('_')[0]
        feature = extract_mfcc(dir+filename).T
        if cnt in rnd:
            if label not in test_dataset.keys():
                test_dataset[label] = []
                test_dataset[label].append(feature)
            else:
                exist_feature = test_dataset[label]
                exist_feature.append(feature)
                test_dataset[label] = exist_feature
        else:
            if label not in train_dataset.keys():
                train_dataset[label] = []
                train_dataset[label].append(feature)
            else:
                exist_feature = train_dataset[label]
                exist_feature.append(feature)
                train_dataset[label] = exist_feature
        if cnt == 50:
            cnt = 1
            rnd = random.sample(range(0, 50), 12)
        else:
            cnt += 1
    return train_dataset, test_dataset

def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    mfcc_features = mfcc(y=wave, sr=sample_rate)
    return mfcc_features

def train_hmm_model(dataset):
    model_dir = 'spoken-digit-hmm-model'
    models = {}
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=10)
        train_data = np.vstack(dataset[label])
        model.fit(train_data)
        models[label] = model
    for label, model in models.items():
        joblib.dump(model, f'{model_dir}/{label}.pkl')
    return models

def load_hmm_models():
    model_dir = 'spoken-digit-hmm-model'
    models = {}
    for pkl in os.listdir(model_dir):
        models[pkl[0]] = joblib.load(f'{model_dir}/{pkl}')
    return models

def train_and_test():
    warnings.filterwarnings('ignore')
    # 1. build dataset from samples
    train_dir = 'spoken_digit_samples/'
    train_dataset, test_dataset = build_dataset(train_dir, rte=0.25)
    # 2. training
    hmm_models = train_hmm_model(train_dataset)
    # 3. predict on test dataset
    acc_count = 0
    all_data_count = 0
    for label in test_dataset.keys():
        feature = test_dataset[label]
        for index in range(len(feature)):
            all_data_count += 1
            scoreList = {}
            for model_label in hmm_models.keys():
                model = hmm_models[model_label]
                score = model.score(feature[index])
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            if predict == label:
                acc_count+=1
    # 4. give out accuracy
    accuracy = round(((acc_count/all_data_count)*100.0), 3)
    print(f'trained model accuracy: {accuracy}')

def predict(audio_path):
    print('running hmm model on `{audio_path}`...')
    models = load_hmm_models()
    feature = extract_mfcc(audio_path).T
    scores = {}
    for (label, model) in models.items():
        score = model.score(feature)
        print(f'score of {label} is {score}')
        scores[label] = score
    predict = max(scores.items(), key=lambda e: e[1])
    print(f'predict result is {predict[0]}, score is {predict[1]}')

if __name__ == '__main__':
    if False:
        train_and_test()
    else:
        import sys
        predict(sys.argv[1])
