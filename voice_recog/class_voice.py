"""
22.05.2018 version 2
"""
# import
import sys
sys.path.append('..')
from voice_recog.funcs_voice import wav_file_in_folder, extract_features, FORMAT, CHANNELS, RATE, CHUNK
import os
import pyaudio
import wave
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


# voice_recognition
class VR:
    def __init__(self, audio, module_path, verbose=False):
        self.modulePath = module_path
        self.verbose = verbose
        # variables
        self.audio = audio
        self.stream = None
        self.frames = []
        self.name = None
        # self.pca = PCA(n_components=10)
        self.rf = None

    def record(self, mode, name=None):
        """
        record a specific length sound and store
        :param name: class name
        :param mode: 0-register 1-predict
        :return:
        """
        seconds = 3
        # set name
        if name:
            self.name = name

        # open mic
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)

        # record specific seconds
        print('Start speak.')
        for i in range(0, int(RATE / CHUNK * seconds)):
            data = self.stream.read(CHUNK)
            self.frames.append(data)

        # close mic
        print('Stop speaking.')
        self.stream.stop_stream()
        self.stream.close()

        # store wav file
        if mode == 0:
            FILE_PATH = os.path.join(os.path.join(os.path.join(self.modulePath, 'train'), self.name), 'sound.wav')
        else:
            FILE_PATH = os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'predict.wav')
        wavf = wave.open(FILE_PATH, 'wb')
        wavf.setnchannels(CHANNELS)
        wavf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wavf.setframerate(RATE)
        wavf.writeframes(b''.join(self.frames))  # append frames recorded to file
        wavf.close()

    def train(self):
        """
        1. find the classes folder
        2. find the wav file in the class
        3. extract features in the wav file
        4. ~perform PCA to reduce dimension~
        5. ~make one-hot labels~
        6. train
        :return: None, store trained model in folder.
        """
        features = []
        labels = []
        for className in os.listdir(os.path.join(self.modulePath, 'train')):
            # if it's not a folder, ignore
            if not os.path.isdir(os.path.join(os.path.join(self.modulePath, 'train'), className)):
                continue
            # get the class folder & search wav file in class
            wavlist = wav_file_in_folder(os.path.join(os.path.join(self.modulePath, 'train'), className))
            if wavlist:
                for wav in wavlist:
                    # get feature of the wav
                    print('className & file: ')
                    print(className)
                    print(wav)
                    feat = extract_features(
                        os.path.join(os.path.join(os.path.join(self.modulePath, 'train'), className), wav))
                    # label = self.one_hot_label(order, classNum)
                    # store feat
                    features.append(feat.flatten())
                    labels.append(className)
                    print(feat.shape)

        # check features & labels
        features = np.array(features)
        labels = np.array(labels)
        print('feature shape: ', features.shape)
        print('label shape: ', labels.shape)

        # train model
        # self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
        # self.knn_clf.fit(features, labels)
        self.rf = RandomForestClassifier(n_estimators=1500, random_state=53)
        self.rf.fit(features, labels)

        # save model
        with open(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf'), 'wb') as f2:
            pickle.dump(self.rf, f2)

    def predict(self):
        # check model
        if not self.rf:
            result = self.open_model()
            # can not open model
            if not result:
                return
        # start predict*****
        try:
            feat = extract_features(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'predict.wav'))
            if feat.shape != (38, 3):
                print('recognize failed! feature length is not correct!')
                return
        except FileNotFoundError as e:
            print('Can not find recorded file! Please record a wav for prediction.')
            return
        closest_distances = self.rf.predict(feat.flatten().reshape(1, -1))  # .reshape(1, -1)
        prob = self.rf.predict_proba(feat.flatten().reshape(1, -1))
        print(closest_distances)
        print(prob)

    def open_model(self):
        self.rf = None
        if os.path.isfile(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf')):
            with open(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf'), 'rb') as f:
                self.rf = pickle.load(f)
            return True
        else:
            print('no voice trained model found!')
            return False

    def on_close(self):
        self.audio.terminate()


if __name__ == '__main__':
    # instantiate the pyaudio
    mic = pyaudio.PyAudio()
    # create instance
    voice = VR(mic, '..')

    # voice.record(0, 'Shan')  # record for registering Shan
    # voice.train()  # create model

    voice.record(1)  # record for prediction
    voice.predict()
    voice.on_close()
