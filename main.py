"""
Main script for face recognition
11.04.2018
"""
# import
import os
import face_recognition
import numpy as np
from face_recog.class_face import recognize_face

# load datasets
data = np.load('src/dataset.npy')
known_face_encodings = list(data[0])
known_face_names = list(data[1])

# run
algo = recognize_face(known_face_encodings, known_face_names)
algo.run()