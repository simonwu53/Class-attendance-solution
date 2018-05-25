# Class-attendance-solution
Course project for pattern recognition. An attendance application for class.


### pre-requisitions
* Python 3+
* tkinter
* sklearn
* numpy

##### face_recognition
* [Installation Guide](https://github.com/ageitgey/face_recognition)
* [Another installation guide](https://www.learnopencv.com/install-dlib-on-macos/)
* dlib(Mac&Linux)
* face_recognition(pip)
* OpenCV
* PIL

##### speech_recognition
* Pyaudio (sound recording)
* librosa (feature extraction)
* wave

##### datasets
* make empty train folder at root before using
* all face pics stored in 'train/' folder
* in 'train/' folder organize all faces by person, each person has a folder named by 'his name'
* structure like this:
```
train/
    person1/
        0.jpg
        1.jpg
        ...
        sound.wav
    person2/
        ...
    ...
```

### RUN
* run 'main.py' in root folder
* ~~(testing) run 'ui.py' in 'gui' folder.~~