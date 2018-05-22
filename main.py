"""
21.05.2018 version 3
"""
# import
import time
import cv2
from gui.ui import ClassAttendanceUI
from face_recog.class_face import FR
from voice_recog.class_voice import VR
import pyaudio

# main
# create face detection app
face = FR("../Class-attendance-solution/",
          n_neighbors=3, verbose=False)

# instantiate the pyAudio
mic = pyaudio.PyAudio()

# create voice detection app
voice = VR(mic, '.')

# create camera stream
print("[INFO] warming up camera...")
videoStream = cv2.VideoCapture(0)
time.sleep(1)

# start main app
app = ClassAttendanceUI(face, voice, videoStream, '../Class-attendance-solution/')
app.on_start()
