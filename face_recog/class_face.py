"""
14.04.2018 version 2
"""
# import
import sys
sys.path.append('../../Class-attendance-solution/face_recog')
from funs_face import train, predict, register_faces
import os
import cv2
import pickle


# face_recognition
class FR:
    def __init__(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        self.verbose = verbose
        self.train_dir = train_dir
        self.model_save_path = model_save_path
        self.n_neighbors = n_neighbors
        self.knn_algo = knn_algo
        self.knn_clf = None
        self.modelPath = '../../Class-attendance-solution/face_recog/trained_knn_model.clf'

    def open_knnclf(self):
        self.knn_clf = None
        if os.path.isfile(self.modelPath):
            with open(self.modelPath, 'rb') as f:
                self.knn_clf = pickle.load(f)
        else:
            print('You should register one face first then start recognition!')
            return

    def train(self):
        print("Training KNN classifier...")
        classifier = train(self.train_dir, model_save_path=self.model_save_path, n_neighbors=self.n_neighbors,
                           verbose=self.verbose)
        print("Training complete!")

    def predict(self, frame):

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(frame, knn_clf=self.knn_clf)

        # Print results on the console
        if self.verbose:
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        # show_prediction_labels_on_image(os.path.join(test_path, image_file), predictions)
        return predictions

    def draw(self, frame, predictions, recover=False):
        for name, (top, right, bottom, left) in predictions:
            if recover:
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame

    def start_recognition(self):
        if not self.knn_clf:
            print('You should register one face first then start recognition!')
            return
        predictitons = None
        process = True
        video_capture = cv2.VideoCapture(0)
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # shrink the size of frame to speed up processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # predict the frame
            if process:
                predictitons = self.predict(small_frame)
            if predictitons:
                frame = self.draw(frame, predictitons, recover=True)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            process = not process

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def register_face(self):
        faces = []
        save_path = self.train_dir
        count = 0
        name = input('Please input your name:')
        if self.verbose:
            print('Your name is: %s' % name)
        os.makedirs(os.path.join(save_path, name))

        video_capture = cv2.VideoCapture(0)  # fps:29
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # save pics
            if count % 8 == 0:
                faces.append(frame)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard or after 3 seconds to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count == 87:
                break

            count += 1

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        status = register_faces(faces, name)
        if self.verbose:
            print('Face registered. Faces captured: %d. Faces registered: %d' % (len(faces), status))
            print('Updating model.')
        # retrain model
        self.train()
        self.open_knnclf()


if __name__ == '__main__':
    f = FR("/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/train", model_save_path="trained_knn_model.clf",
           n_neighbors=3, verbose=False)
    f.open_knnclf()
    f.start_recognition()
    # f.register_face()
    # f.train()
