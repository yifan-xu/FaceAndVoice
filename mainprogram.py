import sys
import json
from os import path

import cv2
import numpy as np


from PyQt5.QtGui import QPixmap, QCursor, QPainter, QImage
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QApplication, QMainWindow
from PyQt5 import QtCore, QtWidgets, QtGui

file_path = "./video1.mp4"
out_path = "./outpy3.avi"
out_size = (200, 200)
data_path = "./out2.json"

class VideoReader:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)
        res, frame = self.cap.read()
        return frame

class VideoWriter:
    def __init__(self, file_path, out_size):
        self.out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M','J','P','G'), 1, out_size)

class DrawFace:
    def __init__(self, data_path):
        self.data = json.load(open(data_path,"r"))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_faces(self, frame, index):
        faces = self.data[str(index)]
        for idx, box in faces.items(): # for box in faces:
            L, T, R, B = box['box']
            cv2.rectangle(frame, (L, T), (R, B), (0,155,255), 2)
            cv2.putText(frame,str(idx),(L, T), self.font, 1,(255,255,255),2,cv2.LINE_AA)

    def generalize_faces(self, box, prev_face=None):
        if prev_face is not None:
            new_overlap = self.get_overlap(box, prev_face)
            if new_overlap > overlap:
                overlap = new_overlap
                max_idx = idx

    def get_overlap(self, r1, r2):
        r1_L, r1_T, r1_R, r1_B = r1['box']
        r2_L, r2_T, r2_R, r2_B = r2['box']
        left = max(r1_L, r2_L)
        right = min(r1_R, r2_R)
        bottom = min(r1_B, r2_B)
        top = max(r1_T, r2_T)
        if left < right and bottom > top:
            intersection = self.get_area(left, right, bottom, top)
            r1_a = getArea(r1_L, r1_R, r1_B, r1_T)
            r2_a = getArea(r2_L, r2_R, r2_B, r2_T)
            overlap = intersection / (r1_a + r2_a - intersection)
        else:
            overlap = 0
        return overlap

    def get_area(self, left, right, bottom, top):
        return (right - left) * (bottom - top)

'''
class RecordVideo(QtCore.QObject):
        image_data = QtCore.pyqtSignal(np.ndarray)

        def __init__(self, camera_port=0, parent=None):
                super().__init__(parent)
                self.camera = cv2.VideoCapture(camera_port)
                self.timer = QtCore.QBasicTimer()

        def start_recording(self):
                self.timer.start(0, self)

        def timerEvent(self, event):
                if (event.timerId() != self.timer.timerId()):
                        return

                read, image = self.camera.read()
                if read:
                        self.image_ready.emit(image)

class FaceDetectionWidget(QtWidgets.QWidget):
        def __init__(self, haar_cascade_filepath, parent=None):
                super().__init__(parent)
                self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
                self.image = QtGui.QImage()
                self._red = (0, 0, 255)
                self._width = 2
                self._min_size = (30, 30)

        def detect_faces(self, image: np.ndarray):
                # haarclassifiers work better in black and white
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.equalizeHist(gray_image)

                faces = self.classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=self._min_size)

                return faces

        def image_data_slot(self, image_data):
                faces = self.detect_faces(image_data)
                for (x, y, w, h) in faces:
                        cv2.rectangle(image_data, (x, y), (x+w, y+h), self._red, self._width)

                self.image = self.get_qimage(image_data)
                if self.image.size() != self.size():
                        self.setFixedSize(self.image.size())

                self.update()

        def get_qimage(self, image: np.ndarray):
                height, width, colors = image.shape
                bytesPerLine = 3 * width
                QImage = QtGui.QImage

                image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

                image = image.rgbSwapped()
                return image

        def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                painter.drawImage(0, 0, self.image)
                self.image = QtGui.QImage()

class MainWidget(QtWidgets.QWidget):
        def __init__(self, haarcascade_filepath, parent=None):
                super().__init__(parent)
                fp = haarcascade_filepath
                self.face_detection_widget = FaceDetectionWidget(fp)

                # TODO: set video port
                self.record_video = RecordVideo()
                self.run_button = QtWidgets.QPushButton('Start')

                # Connect the image data signal and slot together
                image_data_slot = self.face_detection_widget.image_data_slot
                self.record_video.image_data.connect(image_data_slot)
                # connect the run button to the start recording slot
                self.run_button.clicked.connect(self.record_video.start_recording)

                # Create and set the layout
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(self.face_detection_widget)
                layout.addWidget(self.run_button)

                self.setLayout(layout)
'''

class FrameWidget(QWidget):        
    def __init__(self):
        super(FrameWidget, self).__init__()    
        self.image = QtGui.QImage("001.png")
        self.initUI()

    def mousePressEvent(self, QMouseEvent):
        print(QMouseEvent.pos())

    def mouseReleaseEvent(self, QMouseEvent):
        cursor =QCursor()
        #self.statusBar().showMessage('(' + str(QMouseEvent.x()) + ', '+  str(QMouseEvent.y()) + ')')
        print('(', QMouseEvent.x(), ', ', QMouseEvent.y(), ')')
        print(cursor.pos())        

    def initUI(self):                           
        #qbtn = QPushButton('Quit', self)
        #qbtn.resize(qbtn.sizeHint())
        #qbtn.move(50, 50)       
        self.reader = VideoReader(file_path)
        self.faces = DrawFace(data_path)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        #self.statusBar().showMessage('This is a status bar') 

        #label = QLabel(self)


        
        
        '''
        # create painter instance with pixmap
        self.painterInstance = QPainter(pixmap)

        # set rectangle color and thickness
        self.penRectangle = QtGui.QPen(QtCore.Qt.red)
        self.penRectangle.setWidth(3)

        # draw rectangle on painter
        self.painterInstance.setPen(self.penRectangle)
        self.painterInstance.drawRect(590,150,42,55)

        # set pixmap onto the label widget
        #self.ui.label_imageDisplay.setPixmap(pixmap)
        #self.ui.label_imageDisplay.show()
        '''
        #label.setPixmap(pixmap)
        #self.resize(pixmap.width(),pixmap.height())
        #label.move(0, 0)
        #self.show()

    def image_data_slot(self, frame_num):
        frame = self.reader.get_frame(frame_num)
        self.faces.draw_faces(frame, frame_num)

        self.image = self.get_qimage(frame)

        #pixmap = QPixmap(QPixmap.fromImage(image))
        #if self.image.size() != self.size():
        #        self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_widget = FrameWidget()
        self.setGeometry(0, 0, 1024, 768)
        self.setWindowTitle('Quit button')    
        self.run_button = QtWidgets.QPushButton('Start')

        # Connect the image data signal and slot together
        image_data_slot = self.frame_widget.image_data_slot(0)

        # Create and set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.frame_widget)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

def main():        
        app = QApplication(sys.argv)
        main_window = QtWidgets.QMainWindow()
        main_widget = MainWidget()
        main_window.setCentralWidget(main_widget)
        main_window.resize(1024, 768)
        main_window.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
         main()



