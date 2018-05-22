import os
import shutil

# remove models
try:
    os.remove('./face_recog/trained_knn_model.clf')
except IOError as e:
    print('No face recognition model to remove')
try:
    os.remove('./voice_recog/voice_knn_model.clf')
except IOError as e:
    print('No voice recognition model to remove')

# remove logs
for file in os.listdir('./src'):
    if file.endswith('.txt'):
        os.remove(os.path.join('./src', file))

# reset namelist.py  *********
with open('./src/namelist.py', 'w') as f:
    f.write("NAMELIST = ['Kadir', 'Andro']\n")

# remove training classes  *********
for file in os.listdir('./train'):
    if file == 'Shan':
        shutil.rmtree(os.path.join('./train', file))
