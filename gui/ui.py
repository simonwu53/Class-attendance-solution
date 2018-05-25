"""
main logic UI part of the application
"""
import sys

sys.path.append('../../Class-attendance-solution')
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import cv2
from PIL import Image
from PIL import ImageTk
import threading
import time
import os
import pyaudio
from face_recog.class_face import FR
from face_recog.funs_face import register_faces
from voice_recog.class_voice import VR
from src.namelist import NAMELIST


class ClassAttendanceUI:
    def __init__(self, face_detection_module, voice_detection_module, vs, pkg_path):
        # variables
        self.faceDetect = face_detection_module
        self.voiceDetect = voice_detection_module
        self.pkgPath = pkg_path
        self.src = os.path.join(self.pkgPath, 'src')
        self.trainPath = os.path.join(self.pkgPath, 'train')
        self.frame = None
        self.video_stream = vs
        self.faces = []
        self.count = 0
        self.checkedFaces = {}
        self.records = []
        self.font = cv2.FONT_HERSHEY_DUPLEX

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

        self.check_panel = None
        self.submit_button = None
        self.voice_button = None
        self.tree = None
        self.notifyString = StringVar()
        self.notifyString.set('')

        self.msgWin = None
        self.msgWinOpened = False

        # threading loop
        self.isStreaming = None

        # load trained mdoel
        if os.path.isfile(os.path.join(self.pkgPath, 'face_recog/trained_knn_model.clf')) and os.path.isfile(
                os.path.join(self.pkgPath, 'voice_recog/voice_knn_model.clf')):
            self.faceDetect.open_knnclf()
            self.voiceDetect.open_model()
            self.mode = 1  # 1-detection 0-registration 3-no trained data 2-idle mode
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
        self.mode = 2
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
        self.name_entry.focus()
        self.voice_button = Button(self.name_panel, text='Register Voice', command=self.regist_voice)
        self.voice_button.grid(row=2, column=0, sticky=NSEW, padx=230)
        self.submit_button = Button(self.name_panel, text='Submit', command=self.regist_name, state=DISABLED)
        self.submit_button.grid(row=3, column=0, sticky=NSEW, padx=230)
        label = Label(self.name_panel, textvariable=self.notifyString)
        label.grid(row=4, column=0, sticky=NSEW, padx=230)
        # reset msgWin
        if self.msgWinOpened:
            self.msgWin.destroy()
            self.msgWin = None
            self.msgWinOpened = False
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
                try:
                    self.name = predictitons[0][0]  # get name from predictions
                    image = self.faceDetect.draw(self.frame, predictitons, recover=True)
                    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                except IndexError as e:
                    print('No face detected!')
                    self.name = None
                    cv2.putText(small_frame, 'No face detected!', (145, 145), self.font, 1.0, (255, 255, 255), 1)
                    image = small_frame

                # voice detection needed
                if self.name == 'unknown' and not self.msgWinOpened:
                    t = threading.Thread(target=self.messageWindow)
                    t.start()

                # register attendance in background
                if self.name and self.name not in self.checkedFaces and self.name != 'unknown':
                    t = threading.Thread(target=self.logAttendance)
                    t.start()
            elif self.mode == 0:
                if len(self.faces) < 12 and self.count % 15 == 0:
                    print('face captured!')
                    self.faces.append(self.frame)
                # if len of faces get enough, train model
                if len(self.faces) == 12:
                    self.mode = 2  # turn to idle mode
                    # train face & voice model
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

    def check_name(self):
        self.name = self.name_entry.get()
        # check if name is duplicated
        if self.name in NAMELIST:
            # input again
            print('name exist')
            messagebox.showerror("Error", "Name exists!")
            self.name_entry.delete(0, 'end')
            self.name_entry.focus()
            self.name = None
            return False
        else:
            return True

    def regist_name(self):
        result = self.check_name()
        if result:
            self.notifyString.set('')
            # add name to NAMELIST
            NAMELIST.append(self.name)
            # close window
            self.name_panel.destroy()
            # show message box
            messagebox.showinfo("Prepare",
                                "Now you can start register your face, please move your head in a circle loop")
            # start mode 0
            self.mode = 0
            return

    def regist_voice(self):
        result = self.check_name()
        if not result:
            return

        self.voice_button['state'] = 'disabled'
        # make dir for training
        p = os.path.join(self.trainPath, self.name)
        if not os.path.exists(p):
            os.makedirs(p)
        messagebox.showinfo("Prepare",
                            "Now you can start register your voice, please say 'My name is:.....' after a signal")
        t = threading.Thread(target=self.voiceDetect.record, args=(0, self.name, self.notifyString))
        t.start()
        self.submit_button['state'] = 'normal'

    def trainModel(self):
        # store images
        status = register_faces(self.faces, self.name, save_path=self.trainPath)
        print('Face registered. Faces captured: %d. Faces registered: %d' % (len(self.faces), status))
        # update train model
        messagebox.showinfo("Processing", "Training model... Please wait...")
        self.faceDetect.train()
        self.faceDetect.open_knnclf()
        self.voiceDetect.train()
        self.voiceDetect.open_model()
        # log attendance(after training)
        self.logAttendance()
        # back to recognition model
        messagebox.showinfo("Complete", "Now you can start recognition again!")
        self.mode = 1
        return

    def checkNameList(self):
        # check list panel
        self.check_panel = Toplevel(self.root)
        self.tree = ttk.Treeview(self.check_panel, columns=['Checked-in Time'])
        self.tree.heading('#0', text='Name')
        self.tree.heading('#1', text='Checked-in Time')
        self.tree.column('#0', width=100, stretch=YES)
        self.tree.column('#1', stretch=YES)
        self.tree.grid(row=0, column=0, sticky=NSEW)
        refresh_button = Button(self.check_panel, text='Refresh', command=self.refreshTreeView)
        refresh_button.grid(row=1, column=0, sticky=EW)
        # insert data
        self.refreshTreeView()
        return

    def refreshTreeView(self):
        # remove old data
        if self.tree:
            for i in self.tree.get_children():
                self.tree.delete(i)
        # insert data
        for person in self.checkedFaces:
            self.tree.insert('', 'end', text=person, values=(self.checkedFaces[person]))
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
        # record for display
        self.checkedFaces[self.name] = time.strftime("%d-%m-%Y,%H:%M:%S")
        # reset variable
        messagebox.showinfo("Checked in", "your name: %s is checked in!" % self.name)
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

    def recog_voice(self):
        self.mode = 2
        messagebox.showinfo("Prepare", "please say your name ..... after a signal")
        self.voiceDetect.record(1, widget=self. notifyString)
        result = self.voiceDetect.predict()
        if result:
            self.name = result
            self.logAttendance()
        # reset msgWin
        if self.msgWinOpened:
            self.msgWin.destroy()
            self.msgWin = None
            self.msgWinOpened = False
            self.notifyString.set('')
        self.mode = 1

    def messageWindow(self):
        self.msgWinOpened = True
        self.msgWin = Toplevel(self.root)
        self.msgWin.title("Can't recognize you")
        message = "Do u want to register your face & voice or use voice recognition?"
        Label(self.msgWin, text=message).pack()
        Button(self.msgWin, text='Register', command=self.start_registration).pack()
        Button(self.msgWin, text='Voice', command=self.recog_voice).pack()
        Label(self.msgWin, textvariable=self.notifyString).pack()


if __name__ == '__main__':
    # create face detection app
    face = FR("../../Class-attendance-solution/",
              n_neighbors=3, verbose=False)
    # instantiate the pyaudio
    mic = pyaudio.PyAudio()
    # create voice detection app
    voice = VR(mic, '..')
    # create camera stream
    print("[INFO] warming up camera...")
    videoStream = cv2.VideoCapture(0)
    time.sleep(1)
    # start main app
    app = ClassAttendanceUI(face, voice, videoStream, '../../Class-attendance-solution/')
    app.on_start()
