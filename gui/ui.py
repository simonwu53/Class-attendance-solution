import sys

sys.path.append('../../Class-attendance-solution')
from tkinter import *
from tkinter import messagebox
import cv2
from PIL import Image
from PIL import ImageTk
import threading
import time
import os
from face_recog.class_face import FR
from face_recog.funs_face import register_faces
from src.namelist import NAMELIST

"""
TODO: line 93
"""


class ClassAttendanceUI:
    def __init__(self, face_detection_module, vs):
        # variables
        self.faceDetect = face_detection_module
        self.src = '/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/src'
        self.trainPath = '/Users/simonwu/PycharmProjects/PR/Class-attendance-solution/train'
        self.frame = None
        self.video_stream = vs
        self.faces = []
        self.count = 0
        self.checkedFaces = []
        self.records = []

        # ui
        self.root = Tk()
        self.panel = None
        self.detectButton = Button(self.root, text='Detection', command=self.start_detection)
        self.detectButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        self.registButton = Button(self.root, text='Registration', command=self.start_registration)
        self.registButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        self.checkButton = Button(self.root, text='Checklist', command=self.checkNameList)
        self.checkButton.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        self.name_entry = None
        self.name_panel = None
        self.name = None

        # threading loop
        self.isStreaming = None

        # load trained mdoel
        if os.path.isfile('../../Class-attendance-solution/face_recog/trained_knn_model.clf'):
            self.faceDetect.open_knnclf()
            self.mode = 1  # 1-detection 0-registration 3-just video
        else:
            # get in mode 3
            self.mode = 3

        # open log file
        now = time.strftime("%H_%M_%S")
        log_name = 'Log_' + now + '.txt'
        self.logWriter = open(os.path.join(self.src, log_name), 'w')
        self.logWriter.write("timestamp\tname\n")

        # start ui loop
        self.root.wm_title("Class Attendance")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)
        self.video_loop()

    def start_detection(self):
        if self.mode == 3:
            # message box
            messagebox.showerror("Error", "You haven't registered any person!")
            self.start_registration()
            return
        else:
            self.mode = 1
            return

    def start_registration(self):
        # init variable
        self.count = 0
        self.name = None
        self.faces = []

        # register name panel
        self.name_panel = Toplevel(self.root)
        self.name_panel.wm_title("Class Attendance register")
        name_label = Label(self.name_panel, text='Please input your name: ')
        name_label.grid(row=0, column=0, sticky=NSEW, padx=230)
        self.name_entry = Entry(self.name_panel)
        self.name_entry.grid(row=1, column=0, sticky=NSEW, padx=230)
        submit_button = Button(self.name_panel, text='Submit', command=self.regist_name)
        submit_button.grid(row=2, column=0, sticky=NSEW, padx=230)
        return

    def video_loop(self):
        # start displaying video stream
        try:
            ret, self.frame = self.video_stream.read()
            # flip image to right display
            self.frame = cv2.flip(self.frame, 1)
            small_frame = cv2.resize(self.frame, (0, 0), fx=0.5, fy=0.5)
            # check mode
            if self.mode == 1:
                predictitons = self.faceDetect.predict(small_frame)
                self.name = predictitons[0][0]  # get name from predictions
                image = self.faceDetect.draw(self.frame, predictitons, recover=True)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                # register attendance in background ********
                if self.name not in self.checkedFaces:
                    self.checkedFaces.append(self.name)
                    t = threading.Thread(target=self.logAttendance)
                    t.start()
            elif self.mode == 0:
                if len(self.faces) < 12 and self.count % 15 == 0:
                    print('face captured!')
                    self.faces.append(self.frame)
                # if len of faces get enough, train model
                if len(self.faces) == 20:
                    self.mode = 3  # turn to idle mode
                    t = threading.Thread(target=self.trainModel)
                    t.start()
                image = small_frame
                self.count += 1
            else:
                image = small_frame
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
        return

    def regist_name(self):
        self.name = self.name_entry.get()
        # check if name is duplicated
        if self.name in NAMELIST:
            # input again
            print('name exist')
            messagebox.showerror("Error", "Name exists!")
            self.name_entry.delete(0, 'end')
            self.name_entry.focus()
            self.name = None
            return
        else:
            # make dir for training
            p = os.path.join(self.trainPath, self.name)
            os.makedirs(p)
            # add name to NAMELIST
            NAMELIST.append(self.name)
            self.checkedFaces.append(self.name)
            # close window
            self.name_panel.destroy()
            # show message box
            messagebox.showinfo("Prepare",
                                "Now you can start register your face, please move your head in a circle loop")
            # start mode 0
            self.mode = 0
            return

    def trainModel(self):
        # store images
        status = register_faces(self.faces, self.name)
        print('Face registered. Faces captured: %d. Faces registered: %d' % (len(self.faces), status))
        # update train model
        messagebox.showinfo("Processing", "Training model... Please wait...")
        self.faceDetect.train()
        self.faceDetect.open_knnclf()
        # log attendance(after training)
        self.logAttendance()
        # back to recognition model
        self.mode = 1
        return

    def checkNameList(self):
        # use self.records
        return

    def on_start(self):
        print("[INFO] starting...")
        self.root.mainloop()
        return

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.root.after_cancel(self.isStreaming)  # stop camera loop
        self.video_stream.release()  # release camera
        self.isStreaming = None
        self.updateNameList()
        self.logWriter.close()
        self.root.quit()  # quit ui
        return

    def logAttendance(self):
        # write in log file
        now = time.strftime("%d-%m-%Y,%H:%M:%S")
        s = now + '\t' + self.name + '\n'
        self.logWriter.write(s)
        # reset variable
        self.name = None
        return

    def updateNameList(self):
        with open(os.path.join(self.src, 'namelist.py'), 'w') as f:
            s = 'NAMELIST = ['
            for name in NAMELIST:
                s += "'" + name + "'" + ", "
            if len(NAMELIST) > 0:
                s = s[:-2]
            s += "]"
            f.write(s)
        return


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
