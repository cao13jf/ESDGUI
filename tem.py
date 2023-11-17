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
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Overlay")
        self.setGeometry(100, 100, 800, 600)

        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(10, 10, 640, 480)

        self.overlay_button = QPushButton("Toggle Overlay", self)
        self.overlay_button.setGeometry(10, 500, 120, 30)
        self.overlay_button.clicked.connect(self.toggle_overlay)

        self.camera = cv2.VideoCapture(0)  # Replace with the camera index if multiple cameras are available
        self.overlay_enabled = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Set the desired frame update interval in milliseconds

    def update_frame(self):
        ret, frame = self.camera.read()

        if ret:
            if self.overlay_enabled:
                frame = self.overlay_datetime(frame)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap)

    def overlay_datetime(self, frame):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled

if __name__ == "__main__":
    app = QApplication([])
    window = VideoWindow()
    window.show()
    app.exec_()