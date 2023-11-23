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
import cv2

# Open the video file
video = cv2.VideoCapture('dataset/Case_D.MP4')  # Replace with the actual video path

# Get the resolution of the video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Print the resolution
print("Video resolution: {}x{}".format(width, height))

# Release the video file
video.release()