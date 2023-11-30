import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QPainter, QPen
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ContentGenerator(QThread):
    content_generated = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super(ContentGenerator, self).__init__(parent)
        self.running = False

    def run(self):
        self.running = True
        start_time = time.time()

        while self.running:
            current_time = time.time() - start_time
            y_value = random.uniform(1, 4)
            self.content_generated.emit(current_time, y_value)
            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.wait()


class CanvasDrawThread(QThread):
    update_canvas = pyqtSignal()

    def __init__(self, canvas, x_data, y_data, plot_axis, plot_line, parent=None):
        super(CanvasDrawThread, self).__init__(parent)
        self.canvas = canvas
        self.x_data = x_data
        self.y_data = y_data
        self.plot_axis = plot_axis
        self.plot_line = plot_line

    def run(self):
        while True:
            self.update_canvas.emit()
            time.sleep(0.1)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)

        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.start_button.clicked.connect(self.start_content_generation)
        self.stop_button.clicked.connect(self.stop_content_generation)

        self.content_generator = ContentGenerator()
        self.content_generator.content_generated.connect(self.update_data)

        self.x_data = []
        self.y_data = []

        self.plot_axis = None
        self.plot_line = None

        self.canvas_draw_thread = None

        self.closeEvent = self.on_close_event

    def start_content_generation(self):
        self.content_generator.start()
        self.start_canvas_draw_thread()

    def stop_content_generation(self):
        self.content_generator.stop()
        self.stop_canvas_draw_thread()

    def start_canvas_draw_thread(self):
        if self.canvas_draw_thread is None:
            self.canvas_draw_thread = CanvasDrawThread(
                self.canvas, self.x_data, self.y_data, self.plot_axis, self.plot_line
            )
            self.canvas_draw_thread.update_canvas.connect(self.update_canvas)
            self.canvas_draw_thread.start()

    def stop_canvas_draw_thread(self):
        if self.canvas_draw_thread is not None:
            self.canvas_draw_thread.quit()
            self.canvas_draw_thread.wait()
            self.canvas_draw_thread = None

    def update_data(self, x_value, y_value):
        self.x_data.append(x_value)
        self.y_data.append(y_value)

    def update_canvas(self):
        if self.plot_axis is None:
            self.plot_axis = self.figure.add_subplot(111)
            self.plot_line, = self.plot_axis.plot(self.x_data, self.y_data)
        else:
            self.plot_line.set_data(self.x_data, self.y_data)
            self.plot_axis.relim()
            self.plot_axis.autoscale_view()

        self.canvas.draw()

    def on_close_event(self, event):
        self.stop_content_generation()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())