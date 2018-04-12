"""
Analyse faces and build encodings and labels
run this after updated faces
11.04.2018
"""
# import
import os
import face_recognition
import numpy as np

# variables
exclude_file = ['.DS_Store', 'dataset.npy']
src_path = "/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/src"
known_face_encodings = []
known_face_names = []

# load datasets
file_walk = os.walk(src_path)
for path, dir_list, file_list in file_walk:
    for file_name in file_list:
        if file_name not in exclude_file:
            image = face_recognition.load_image_file(os.path.join(path, file_name))
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(path.split('/')[-1])

np.save('src/dataset.npy', [known_face_encodings, known_face_names])
