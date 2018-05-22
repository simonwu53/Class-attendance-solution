"""
22.05.2018 version 2
"""
# import
import sys
sys.path.append('..')
from voice_recog.funcs_voice import wav_file_in_folder, FORMAT, CHANNELS, RATE, CHUNK
import os
from array import array
import pyaudio
import wave
import librosa
import numpy as np
from sklearn import neighbors
import pickle
from sklearn.decomposition import PCA


# voice_recognition
class VR:
    def __init__(self, audio, module_path, verbose=False):
        self.modulePath = module_path
        self.verbose = verbose
        self.filename_suffix = '.wav'
        # variables
        self.audio = audio
        self.stream = None
        self.frames = []
        self.name = None
        # self.namelist = {}
        # self.pca = PCA(n_components=8)
        self.knn_clf = None

    def record(self, mode, seconds, name=None):
        """
        record a specific length sound and store
        :param name: class name
        :param mode: 0-register 1-predict
        :return:
        """
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
            # # split voice array
            # partition = np.floor(len(self.frames)/3)
            # part_1 = self.frames[:partition]
            # part_2 = self.frames[partition:partition*2]
            # part_3 = self.frames[partition*2:]
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
        order = 0
        for className in os.listdir(os.path.join(self.modulePath, 'train')):
            # if it's not a folder, ignore
            if not os.path.isdir(os.path.join(os.path.join(self.modulePath, 'train'), className)):
                continue
            # get the class folder & search wav file in class
            # self.namelist[order] = className
            wavlist = wav_file_in_folder(os.path.join(os.path.join(self.modulePath, 'train'), className))
            if wavlist:
                for wav in wavlist:
                    # get feature of the wav
                    print('className & file: ')
                    print(className)
                    print(wav)
                    feat = self.extract_features(
                        os.path.join(os.path.join(os.path.join(self.modulePath, 'train'), className), wav))
                    # store feat
                    features.append(feat.flatten())
                    labels.append(className)
            order += 1
        # perform pca
        # features = self.pca.fit_transform(features)
        # print(features.shape)
        # create labels
        # classNum = len(self.namelist)
        # for key in self.namelist:
        #     labels.append(self.one_hot_label(key, classNum))

        print('labels: ', labels)

        # check features & labels
        features = np.array(features)
        labels = np.array(labels)
        print('feature shape: ', features.shape)
        print('label shape: ', labels.shape)

        # save name list for later use
        # print(self.namelist)
        # np.save('namelist.npy', self.namelist)

        # train model
        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
        self.knn_clf.fit(features, labels)

        # save model
        with open(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf'), 'wb') as f:
            pickle.dump(self.knn_clf, f)

    def predict(self):
        # check model
        if not self.knn_clf:
            result = self.open_knnclf()
            # can not open model
            if not result:
                return
        # start predict*****
        try:
            feat = self.extract_features(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'predict.wav'))
        except FileNotFoundError as e:
            print('Can not find recorded file! Please record a wav for prediction.')
            return
        closest_distances = self.knn_clf.predict(feat.flatten().reshape(1, -1))
        print(closest_distances)

    @staticmethod
    def extract_features(wav):
        """
        function to extract wav features
        :param wav: path to wav file
        :return: mfcc features
        """
        y, sr = librosa.load(wav, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
        return np.concatenate((mfcc, mfcc_delta), axis=1)

    @staticmethod
    def one_hot_label(order, num_classes=10):
        """
        crate one-hot label
        :param order: the class order
        :param num_classes: number of classes
        :return: one-hot label
        """
        return np.eye(num_classes)[order]

    def open_knnclf(self):
        self.knn_clf = None
        if os.path.isfile(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf')):
            with open(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf'), 'rb') as f:
                self.knn_clf = pickle.load(f)
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

    # voice.record(0, 9, 'Andro')  # record for registering Shan
    # voice.train()  # create model

    voice.record(1, 3)  # record for prediction
    voice.predict()
    voice.on_close()
