"""
21.05.2018 version 1
"""
# import
import os
import sys
import pyaudio
sys.path.append('../../Class-attendance-solution/')

# static variables
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024


# static funcs
def wav_file_in_folder(path):
    """
    make a list of wave file in the folder
    :param path: folder want to search
    :return: a list of wav file, None if no wav file
    """
    file_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.wav':
            file_list.append(file)
    if len(file_list) == 0:
        return None
    else:
        return file_list


if __name__ == '__main__':
    wav_file_in_folder('/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/voice_recog')
    wav_file_in_folder('/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/train/Stella')
