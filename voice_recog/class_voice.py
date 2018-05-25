"""
25.05.2018 version 3
add widget to notify user to start/stop speaking
"""
# import
import sys
sys.path.append('..')
from voice_recog.funcs_voice import wav_file_in_folder, extract_features, play_sound, FORMAT, CHANNELS, RATE, CHUNK
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
        self.rf = None
        self.labels = None

    def record(self, mode, name=None, widget=None):
        """
        record a specific length sound and store
        :param widget: widget from ui for notifying user to speak
        :param name: class name
        :param mode: 0-register 1-predict
        :return: save wav file in train folder
        """
        seconds = 3
        # set name
        if name:
            self.name = name

        # notify starting
        if widget:
            widget.set('Start speaking...')

        # open mic & record specific seconds
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
        print('Start speaking.')
        for i in range(0, int(RATE / CHUNK * seconds)):
            data = self.stream.read(CHUNK)
            self.frames.append(data)

        # close mic
        print('Stop speaking.')
        self.stream.stop_stream()
        self.stream.close()
        if widget:
            widget.set('Stop speaking...')

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
        self.labels = []
        print('Training voice dataset...')
        for className in os.listdir(os.path.join(self.modulePath, 'train')):
            # if it's not a folder, ignore
            if not os.path.isdir(os.path.join(os.path.join(self.modulePath, 'train'), className)):
                continue
            # get the class folder & search wav file in class
            wavlist = wav_file_in_folder(os.path.join(os.path.join(self.modulePath, 'train'), className))
            if wavlist:
                for wav in wavlist:
                    # get feature of the wav
                    feat = extract_features(
                        os.path.join(os.path.join(os.path.join(self.modulePath, 'train'), className), wav))
                    # store feat
                    # if feat.shape != (38, 3):
                    #     print("%s 's voice didn't trained, please record again." % className)
                    #     continue
                    features.append(feat.flatten())
                    self.labels.append(className)

        # check features & labels
        features = np.array(features)
        self.labels = np.array(self.labels)
        print('feature shape: ', features.shape)
        print('label shape: ', self.labels.shape)

        # train model
        # self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
        # self.knn_clf.fit(features, labels)
        self.rf = RandomForestClassifier(n_estimators=1500, random_state=53)
        self.rf.fit(features, self.labels)

        # save model
        with open(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'voice_knn_model.clf'), 'wb') as f2:
            pickle.dump(self.rf, f2)
            print('Training complete.')

    def predict(self):
        # check model
        if not self.rf:
            result = self.open_model()
            # can not open model
            if not result:
                return
        # start predict
        try:
            feat = extract_features(os.path.join(os.path.join(self.modulePath, 'voice_recog'), 'predict.wav'))
            # if feat.shape != (38, 3):
            #     print('recognize failed! feature length is not correct!')
            #     return False
        except FileNotFoundError as e:
            print('Can not find recorded file! Please record a wav for prediction.')
            return False
        pred = self.rf.predict(feat.flatten().reshape(1, -1))  # .reshape(1, -1)
        prob = self.rf.predict_proba(feat.flatten().reshape(1, -1))
        print(prob)
        threshold = 1 / len(prob[0])
        prob = max(prob[0]) > threshold
        if prob:
            return pred
        else:
            return False

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

    # voice.record(0, 'Ayane')  # record for registering Shan
    voice.train()  # create model

    # voice.record(1)  # record for prediction
    # print(voice.predict())
    voice.on_close()
