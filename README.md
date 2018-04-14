# Class-attendance-solution
Course project for pattern recognition. An attendance application for class.


### pre-requisitions

##### face_recognition
* [Installation Guide](https://github.com/ageitgey/face_recognition)
* [Another installation guide](https://www.learnopencv.com/install-dlib-on-macos/)
* dlib(Mac&Linux)
* face_recognition(pip)
* OpenCV

##### datasets
* all face pics stored in 'src/' folder
* in 'src/' folder organize all faces by person, each person has a folder named by 'his name'
* structure like this:
'''
src/
    person1/
        0.jpg
        1.jpg
        ...
    person2/
        ...
    ...
'''
* 'src2/' has the same structure, but all pics are cropped to show only faces.

### RUN
* put pics of persons in src/ by instructions above.
* run 'crop_faces.py' to create dataset only contains faces(faces in src2/).
* run 'build_dataset.py' to prepare database for recognition.
* run 'main.py' to play.