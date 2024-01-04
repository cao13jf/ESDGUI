import os
import cv2
import time
import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar

from utils.guis import PhaseCom
from utils.report_tools import generate_report


class ImageProcessingThread(QThread):
    processed_frame = pyqtSignal(np.ndarray)
    processing_stop_signal = pyqtSignal()
    def __init__(self, start_x, end_x, start_y, end_y, cfg):
        super().__init__()
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.frames_to_process = []
        self.frames_to_process_len = 0
        self.phaseseg = PhaseCom(arg=cfg)
        self.processing_interval = 2  # Control the
        self.processing_stop = False

    def run(self):
        while not self.processing_stop:
            if self.frames_to_process_len >= 1:
                frame = self.frames_to_process
                pred = self.process_image(frame)
                self.frames_to_process = []
                self.frames_to_process_len = 0
                self.processed_frame.emit(pred)
            if self.processing_stop:
                break

    def process_image(self, cv_img):
        # cv_img = img[self.start_x:self.end_x, self.start_y:self.end_y]  # Crop images
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pred = self.phaseseg.phase_frame(rgb_image)

        return pred

    def add_frame(self, frame):
        self.frames_to_process = frame
        self.frames_to_process_len += 1

    def stop(self):
        self.processing_stop = True
        self.processing_stop_signal.emit()

class VideoReadThread(QThread):
    frame_data = pyqtSignal(np.ndarray, int)

    def __init__(self):
        super().__init__()
        self._is_running = False
        self.frame_index = 0

    def run(self):
        # time.sleep(30)
        self._is_running = True
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # camera.set(cv2.CAP_PROP_FPS, 50)

        while self._is_running:
            ret, frame = camera.read()
            if ret:
                self.frame_index += 1
                self.frame_data.emit(frame, self.frame_index)

            if not ret:
                camera.release()
                self.frame_index = 0
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                camera.set(cv2.CAP_PROP_FPS, 17)
                continue

        camera.release()


class ReportThread(QThread):
    out_file_path = pyqtSignal(str)

    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.update_report = False

    def run(self):
        while True:
            if self.update_report:

                dialog = QDialog()
                dialog.setWindowTitle("Progress")
                layout = QVBoxLayout(dialog)
                progress_label = QLabel("Progress: 0%")
                progress_bar = QProgressBar()
                layout.addWidget(progress_label)
                layout.addWidget(progress_bar)
                dialog.setFixedSize(300, 100)
                dialog.show()

                report_file = generate_report(self.log_dir, progress_label, progress_bar, dialog)
                report_path = os.path.abspath(report_file)
                QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
                self.update_report = False
                self.out_file_path.emit(report_file)
    def report_signal(self, generate_flag):
        self.update_report = generate_flag
        print("Geting report signal", generate_flag)

class PlotCurveThread(QThread):
    ploted_curve_array = pyqtSignal(np.ndarray)
    plot_stop_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.processing_interval = 10
        self.data_to_plot = []
        self.data_to_plot_len = 0
        self.plot_stop=False

    def run(self):
        prev_time = int(time.time())
        while not self.plot_stop:
            cur_time = int(time.time())
            if cur_time > prev_time and self.data_to_plot_len >= self.processing_interval:
                prev_time = cur_time
                cur_data = self.data_to_plot[0]
                time_len = self.data_to_plot[1] + 2
                self.data_to_plot = []
                self.data_to_plot_len = 0
                fig = Figure(figsize=(5.7, 3.4), dpi=80)
                fig.patch.set_facecolor('lightgray')
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(np.linspace(1, time_len, cur_data.shape[0]) / 60, cur_data, linewidth=2)
                ax.set_xlabel("Time / Minute")
                ax.set_ylabel("Normalized Transition Index")
                ax.grid(True, axis='y')
                ax.set_facecolor('lightgray')
                ax.set_xlim(0, 8)
                ax.set_ylim(-0.01, max(np.max(cur_data) * 5 / 4, 0.1))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                # Render the figure to a RGB array
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

                self.ploted_curve_array.emit(image)
            if self.plot_stop:
                break

    def add_data(self, data):
        self.data_to_plot = data
        self.data_to_plot_len += 1

    def stop(self):
        self.plot_stop = True
        self.plot_stop_signal.emit()
