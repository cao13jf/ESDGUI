#
#
# import sys
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
# from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtGui import QPainter, QFont
# import cv2
#
#
# class CameraThread(QThread):
#     frame_processed = pyqtSignal(np.ndarray)
#
#     def run(self):
#         camera = cv2.VideoCapture(0)
#         while True:
#             ret, frame = camera.read()
#             if not ret:
#                 break
#
#             processed_frame = cv2.Canny(frame, 100, 200)  # Apply Canny edge detection
#             self.frame_processed.emit(processed_frame)
#
#         camera.release()
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         # Initialize the data variables
#         self.time_elapsed = []
#         self.status_idle = []
#         self.status_phase = []
#         self.video_width = 400
#
#         # Set up the GUI components
#         self.setWindowTitle("Dynamic Plot Demo")
#         self.canvas_idle = FigureCanvas(Figure(figsize=(400/80, 200/80), dpi=80))
#         self.canvas_phase = FigureCanvas(Figure(figsize=(400/80, 50/80), dpi=80))
#         self.start_button = QPushButton("Start")
#         self.stop_button = QPushButton("Stop")
#         self.camera_label = QLabel()
#
#         # Set the layout
#         central_widget = QWidget()
#         layout = QVBoxLayout(central_widget)
#         layout.addWidget(self.canvas_idle)
#         layout.addWidget(self.canvas_phase)
#         layout.addWidget(self.start_button)
#         layout.addWidget(self.stop_button)
#         layout.addWidget(self.camera_label)  # Add the camera label to the layout
#         self.setCentralWidget(central_widget)
#
#         # Connect button signals
#         self.start_button.clicked.connect(self.start_updates)
#         self.stop_button.clicked.connect(self.stop_updates)
#
#         # Set up the timer for dynamic updates
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_data)
#
#         # Initialize the plots
#         self.ax_idle = self.canvas_idle.figure.subplots()
#         self.ax_idle.set_xlabel("Time (seconds)")
#         self.ax_idle.set_ylabel("Idle Status")
#
#         self.ax_phase = self.canvas_phase.figure.subplots()
#         self.ax_phase.set_axis_off()
#         self.phase_colors = {"idle": "blue", "marking": "green", "injection": "yellow", "dissection": "red"}
#
#         # Initialize the camera thread
#         self.camera_thread = CameraThread()
#         self.camera_thread.frame_processed.connect(self.update_camera_frame)
#
#     def start_updates(self):
#         self.timer.start(100)  # Update every second (1000 milliseconds)
#         self.camera_thread.start()  # Start the camera thread
#
#     def stop_updates(self):
#         self.timer.stop()
#         self.camera_thread.terminate()  # Terminate the camera thread
#
#     def update_data(self):
#         # Generate random system status
#         status = random.choice(["idle", "marking", "injection", "dissection"])
#
#         # Update the data lists
#         self.status_phase.append(status)
#         self.time_elapsed.append(len(self.time_elapsed) + 1)
#         self.status_idle.append(self.status_phase.count("idle"))
#
#         # Update the idle status plot
#         self.ax_idle.plot(self.time_elapsed, self.status_idle, color="blue")
#         self.ax_idle.yaxis.grid(True)  # Set y-grid on
#         self.ax_idle.xaxis.grid(False)  # Set x-grid off
#         self.ax_idle.set_xlim(1, len(self.time_elapsed) * 6 / 5)
#
#         # Update the phase color bar
#         self.ax_phase.clear()
#         self.ax_phase.set_xlim(0, len(self.time_elapsed) * 6 / 5)
#         self.ax_phase.set_ylim(0, 1)
#
#         # Create a color map to map phase names to colors
#         cmap = plt.get_cmap("Set1", len(self.phase_colors))
#
#         # Create an array of colors for the current status
#         colors = np.array([cmap(i) for i in range(len(self.phase_colors))])
#
#         # Create an image plot with the colors
#         for i, phase in enumerate(self.status_phase):
#             self.ax_phase.bar(i, 1, color=self.phase_colors[phase], width=1.0)
#
#         self.canvas_idle.draw()
#         self.canvas_phase.draw()
#
#     def update_camera_frame(self, processed_frame):
#         # Convert the processed frame to QImage
#         image = QImage(
#             processed_frame.data, processed_frame.shape[1], processed_frame.shape[0],
#             processed_frame.strides[0], QImage.Format_Grayscale8
#         )
#
#         # Convert QImage to QPixmap for displaying in QLabel
#         pixmap = QPixmap.fromImage(image)
#
#         # Scale the pixmap to fit the label dimensions
#         scaled_pixmap = pixmap.scaled(
#             self.video_width, self.camera_label.height(), Qt.KeepAspectRatio
#         )
#
#         # Create a new pixmap with the same size as the scaled pixmap
#         combined_pixmap = QPixmap(scaled_pixmap.size())
#         combined_pixmap.fill(Qt.transparent)
#
#         # Create a painter object for drawing on the combined pixmap
#         painter = QPainter(combined_pixmap)
#         painter.drawPixmap(0, 0, scaled_pixmap)  # Draw the scaled video pixmap
#
#         # Set text color and font
#         painter.setPen(Qt.white)
#         font_size = int(self.camera_label.height() / 10)
#         font = QFont("Arial", font_size)
#         painter.setFont(font)
#
#         # Calculate the position to draw the time text
#         current_time = str(len(self.time_elapsed))
#         text_width = painter.fontMetrics().width(current_time)
#         text_height = painter.fontMetrics().height()
#         text_x = combined_pixmap.width() - text_width - 10
#         text_y = combined_pixmap.height() - text_height - 10
#
#         # Draw the time text on the combined pixmap
#         painter.drawText(text_x, text_y, current_time)
#
#         # End painting
#         painter.end()
#
#         # Set the combined pixmap as the label's pixmap
#         self.camera_label.setPixmap(combined_pixmap)
#
#         # Update the layout to reflect the changes
#         self.camera_label.update()
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
#
#
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
import cv2


class CameraThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def run(self):
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            processed_frame = cv2.Canny(frame, 100, 200)  # Apply Canny edge detection
            self.frame_processed.emit(processed_frame)

        camera.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the data variables
        self.time_elapsed = []
        self.status_idle = []
        self.status_phase = []

        # Set up the GUI components
        self.setWindowTitle("Dynamic Plot Demo")
        self.canvas_idle = FigureCanvas(Figure(figsize=(4, 2), dpi=80))
        self.canvas_phase = FigureCanvas(Figure(figsize=(4, 1), dpi=80))
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.camera_label = QLabel()

        # Set the layout
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)

        # Left part of the layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas_idle)
        left_layout.addWidget(self.canvas_phase)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)
        layout.addLayout(left_layout)

        # Right part of the layout
        layout.addWidget(self.camera_label)

        self.setCentralWidget(central_widget)

        # Connect button signals
        self.start_button.clicked.connect(self.start_updates)
        self.stop_button.clicked.connect(self.stop_updates)

        # Set up the timer for dynamic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)

        # Initialize the plots
        self.ax_idle = self.canvas_idle.figure.subplots()
        self.ax_idle.set_xlabel("Time (seconds)")
        self.ax_idle.set_ylabel("Idle Status")

        self.ax_phase = self.canvas_phase.figure.subplots()
        self.ax_phase.set_axis_off()
        self.phase_colors = {"idle": "blue", "marking": "green", "injection": "yellow", "dissection": "red"}

        # Initialize the camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_processed.connect(self.update_camera_frame)

    def start_updates(self):
        self.timer.start(100)  # Update every second (1000 milliseconds)
        self.camera_thread.start()  # Start the camera thread

    def stop_updates(self):
        self.timer.stop()
        self.camera_thread.terminate()  # Terminate the camera thread

    def update_data(self):
        # Generate random system status
        status = random.choice(["idle", "marking", "injection", "dissection"])

        # Update the data lists
        self.status_phase.append(status)
        self.time_elapsed.append(len(self.time_elapsed) + 1)
        self.status_idle.append(self.status_phase.count("idle"))

        # Update the idle status plot
        self.ax_idle.plot(self.time_elapsed, self.status_idle, color="blue")
        self.ax_idle.yaxis.grid(True)  # Set y-grid on
        self.ax_idle.xaxis.grid(False)  # Set x-grid off
        self.ax_idle.set_xlim(1, len(self.time_elapsed) * 6 / 5)

        # Update the phase color bar
        self.ax_phase.clear()
        self.ax_phase.set_xlim(0, len(self.time_elapsed) * 6 / 5)
        self.ax_phase.set_ylim(0, 1)

        # Create a color map to map phase names to colors
        cmap = plt.get_cmap("Set1", len(self.phase_colors))

        # Create an array of colors for the current status
        colors = np.array([cmap(i) for i in range(len(self.phase_colors))])

        # Create an image plot with the colors
        for i, phase in enumerate(self.status_phase):
            self.ax_phase.bar(i, 1, color=self.phase_colors[phase])

        self.canvas_idle.draw()
        self.canvas_phase.draw()

    def resizeEvent(self, event):
        # Override the resizeEvent to resize the video frame
        size = self.camera_label.size()
        self.camera_label.setPixmap(self.scaled_pixmap.scaled(size, Qt.KeepAspectRatio))

    def update_camera_frame(self, frame):
        # Convert the frame to QImage
        height, width = frame.shape
        q_image = QImage(frame.data, width, height, QImage.Format_Grayscale8)

        # Create a QPixmap from QImage
        pixmap = QPixmap.fromImage(q_image)

        # Scale the QPixmap to fit the label size
        size = self.camera_label.size()
        self.scaled_pixmap = pixmap.scaled(size, Qt.KeepAspectRatio)

        # Set the scaled QPixmap as the label's pixmap
        self.camera_label.setPixmap(self.scaled_pixmap)

    def resizeEvent(self, event):
        # Override the resizeEvent to resize the video frame
        if hasattr(self, 'scaled_pixmap'):
            size = self.camera_label.size()
            self.camera_label.setPixmap(self.scaled_pixmap.scaled(size, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.stop_updates()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())