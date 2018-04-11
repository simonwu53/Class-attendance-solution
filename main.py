"""
Wrote by Shan
11.04.2018
"""
# import
import os
import face_recognition
from face_recog.class_face import recognize_face

# variables
exclude_file = '.DS_Store'
src_path = "/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/src"
known_face_encodings = []
known_face_names = []

# load datasets
file_walk = os.walk(src_path)
for path, dir_list, file_list in file_walk:
    for file_name in file_list:
        if file_name != exclude_file:
            image = face_recognition.load_image_file(os.path.join(path, file_name))
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(path.split('/')[-1])
# run
algo = recognize_face(known_face_encodings, known_face_names)
algo.run()