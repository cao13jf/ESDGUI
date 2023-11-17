# ======================
# demo for reading camera
# ======================

# # import the opencv library
# import cv2
#
# # define a video capture object
# vid = cv2.VideoCapture(0)
#
# while (True):
#
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()


# =========================
# read
# =========================
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QCoreApplication


class ImageProcessingThread(QThread):
    processed_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frames_to_process = []
        self.processing_interval = 5

    def run(self):
        while True:
            if len(self.frames_to_process) >= self.processing_interval:
                frame = self.frames_to_process.pop(0)
                processed_frame = self.process_image(frame)
                self.processed_frame.emit(processed_frame)
            self.msleep(1)  # Sleep for a short duration to avoid high CPU usage

    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        pred = cv2.addWeighted(frame, 0.8, colored_edges, 1, 0)
        return processed_frame

    def add_frame(self, frame):
        self.frames_to_process.append(frame)


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Display")
        self.setGeometry(100, 100, 800, 600)

        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(10, 10, 640, 480)

        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(10, 500, 120, 30)
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(140, 500, 120, 30)
        self.stop_button.clicked.connect(self.stop_processing)

        self.camera = cv2.VideoCapture(0)
        self.process_frames = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.processing_thread = ImageProcessingThread()
        self.processing_thread.processed_frame.connect(self.display_processed_frame)
        self.processing_thread.moveToThread(QCoreApplication.instance().thread())
        self.processing_thread.start()

    def update_frame(self):
        ret, frame = self.camera.read()

        if ret:
            if self.process_frames:
                self.processing_thread.add_frame(frame)
            else:
                self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.camera_label.setPixmap(pixmap)

    def display_processed_frame(self, processed_frame):
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.camera_label.setPixmap(pixmap)

    def start_processing(self):
        self.process_frames = True

    def stop_processing(self):
        self.process_frames = False

    def closeEvent(self, event):
        self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())