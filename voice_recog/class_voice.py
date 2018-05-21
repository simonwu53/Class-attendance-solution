"""
21.05.2018 version 1
"""
# import
import sys
sys.path.append('..')
from voice_recog.funcs_voice import wav_file_in_folder, FORMAT, CHANNELS, RATE, CHUNK
import os
import cv2
import pickle


# voice_recognition
class VR:
    def __init__(self, module_path, verbose=False):
        self.modulePath = module_path
        self.verbose = verbose
        self.filename_suffix = '.wav'
        self.RECORD_SECONDS = 5


if __name__ == '__main__':
    voice = VR('..')