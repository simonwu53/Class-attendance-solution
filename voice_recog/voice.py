import pyaudio
import sys
import wave
import time
from array import array
from scipy.io import wavfile
import numpy as np

FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
CHUNK=1024
RECORD_SECONDS=5
FILE_NAME="RECORDING.wav"

audio=pyaudio.PyAudio() #instantiate the pyaudio

#recording prerequisites
stream=audio.open(format=FORMAT,channels=CHANNELS, 
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

#starting recording
frames=[]

start = time.time()
for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    data=stream.read(CHUNK)
    data_chunk=array('h',data)
    vol=max(data_chunk)
    if(vol>=500):
        print("something said")
        frames.append(data)
    else:
        print("nothing")
    print("\n")

end = time.time()

#end of recording
stream.stop_stream()
stream.close()
audio.terminate()
#writing to file
wavfile1=wave.open(FILE_NAME,'wb')
wavfile1.setnchannels(CHANNELS)
wavfile1.setsampwidth(audio.get_sample_size(FORMAT))
wavfile1.setframerate(RATE)
wavfile1.writeframes(b''.join(frames))#append frames recorded to file
wavfile1.close()
print("done - result written to RECORDING.wav")
fs, voice_array = wavfile.read('RECORDING.wav')
voice_array= np.array(voice_array)
print("Shape of voice_array =",voice_array.shape)    
print("voice_array= ",voice_array)
print("Mean of voice_array =",np.mean(voice_array))