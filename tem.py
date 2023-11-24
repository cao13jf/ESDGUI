import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the data variables
        self.time_elapsed = []
        self.status_idle = []
        self.status_phase = []

        # Set up the GUI components
        self.setWindowTitle("Dynamic Plot Demo")
        self.canvas_idle = FigureCanvas(Figure())
        self.canvas_phase = FigureCanvas(Figure())
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        # Set the layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas_idle)
        layout.addWidget(self.canvas_phase)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
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

    def start_updates(self):
        self.timer.start(100)  # Update every second (1000 milliseconds)

    def stop_updates(self):
        self.timer.stop()

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
            self.ax_phase.bar(i, 1, color=self.phase_colors[phase], width=1.0)

        self.canvas_idle.draw()
        self.canvas_phase.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())