import sys
sys.path.append('../../Class-attendance-solution')
from tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk
import threading
import time
import os
# from face_recog.funs_face import predict, draw
from face_recog.class_face import FR


class ClassAttendanceUI:
    def __init__(self, face_detection_module, vs):
        # variables
        self.faceDetect = face_detection_module
        self.src = '/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/src'
        self.frame = None
        self.video_stream = vs
        self.faces = []
        self.count = 0
        self.registName = None
        if os.path.isfile('../../Class-attendance-solution/face_recog/trained_knn_model.clf'):
            self.faceDetect.open_knnclf()
            self.mode = 1  # 1-detection 0-registration
        else:
            # start registration
            # show dialog click button to start register
            self.mode = 0  # put this in the register

        # ui
        self.root = Tk()
        self.panel = None
        self.detectButton = Button(self.root, text='Detection', command=self.start_detection)
        self.detectButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        self.registButton = Button(self.root, text='Registration', command=self.start_registration)
        self.registButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # threading
        self.isStreaming = None

        # start ui loop
        self.root.wm_title("Class Attendance")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)
        self.video_loop()

    def start_detection(self):
        self.mode = 1

    def start_registration(self):
        # init variable
        self.count = 0
        self.registName = None
        # register name
        self.regist_name()

    def video_loop(self):
        # start displaying video stream
        try:
            ret, self.frame = self.video_stream.read()
            small_frame = cv2.resize(self.frame, (0, 0), fx=0.5, fy=0.5)
            # check mode
            if self.mode:
                predictitons = self.faceDetect.predict(small_frame)
                image = self.faceDetect.draw(self.frame, predictitons, recover=True)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            else:
                if len(self.faces) < 10 & self.count % 5 == 0:
                    self.faces.append(self.frame)
                image = small_frame
                self.count += 1
            # change frame to PIL RGB format and then convert to ImageTK
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            # if the panel is None, we need to initialize it
            if self.panel is None:
                self.panel = Label(image=image)
                self.panel.image = image
                self.panel.pack(side="left", padx=10, pady=10)
            else:
                # else just update the image frame
                self.panel.configure(image=image)
                self.panel.image = image
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")
        self.isStreaming = self.root.after(25, self.video_loop)

    def regist_name(self):
        # set registName
        # click button to start register mode
        pass

    def on_start(self):
        print("[INFO] starting...")
        self.root.mainloop()

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.root.after_cancel(self.isStreaming)  # stop camera loop
        self.video_stream.release()  # release camera
        self.isStreaming = None
        self.root.quit()  # quit ui


if __name__ == '__main__':
    # create face detection app
    face = FR("/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/train",
              model_save_path="/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/face_recog/trained_knn_model.clf",
              n_neighbors=3, verbose=False)
    # create voice detection app
    # create camera stream
    print("[INFO] warming up camera...")
    videoStream = cv2.VideoCapture(0)
    time.sleep(1)
    # start main app
    app = ClassAttendanceUI(face, videoStream)
    app.on_start()
