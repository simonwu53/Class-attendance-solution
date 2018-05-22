"""
21.05.2018 version 1
"""
# import
import os
import pyaudio
import numpy as np
import librosa
import wave

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


def extract_features(wav):
    """
    function to extract wav features
    :param wav: path to wav file
    :return: beat-synchronous features
    """
    # y, sr = librosa.load(wav, mono=True)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # mfcc_delta = librosa.feature.delta(mfcc)
    # return np.concatenate((mfcc, mfcc_delta), axis=1)

    # Load the example clip
    y, sr = librosa.load(wav)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    return beat_features


def play_sound(audio, path):  # use audio object from an instance
    # length of data to read.
    chunk = 1024
    # open the file for reading.
    wf = wave.open(path, 'rb')

    # create an audio object
    # p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = audio.open(format=
                        audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while len(data) > 0:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    wav_file_in_folder('/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/voice_recog')
    wav_file_in_folder('/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/train/Stella')
    # play_sound()
