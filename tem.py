# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
#
# # Generate some data
# x = np.linspace(0, 2 * np.pi, 100)
# y = np.sin(x)
#
# # Create a figure and plot the data
# fig = Figure()
# canvas = FigureCanvas(fig)
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x, y)
#
# # Render the figure to a RGB array
# canvas.draw()
# width, height = fig.get_size_inches() * fig.get_dpi()
# image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
#
# # Display the shape of the generated image array
# print(image.shape)


from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import concatenate_videoclips
from moviepy.editor import VideoFileClip
from datetime import time

# Define the start and end times for each clip in the format of hour-minute-second
# idle 00:00:50-00:03:54
# marking 00:10:50-00:14:30
# injection 00:19:15-00:21:10
# dissection 00:21:10-00:25:00
#
# clips = [
#     # {"start_time": time(hour=0, minute=2, second=30), "end_time": time(hour=0, minute=2, second=55)},
#     {"start_time": time(hour=0, minute=10, second=50), "end_time": time(hour=0, minute=14, second=30)},
#     {"start_time": time(hour=0, minute=19, second=15), "end_time": time(hour=0, minute=21, second=10)},
#     {"start_time": time(hour=0, minute=21, second=10), "end_time": time(hour=0, minute=25, second=0)}
#     # Add more clips as needed
# ]
#
# # Convert the time to seconds
# # Convert the time to seconds
# def time_to_seconds(t):
#     return t.hour * 3600 + t.minute * 60 + t.second
#
# extracted_clips = []
# video_path = "dataset/Case_D.MP4"  # Replace with the path to your video file
#
# for clip in clips:
#     start_time = time_to_seconds(clip["start_time"])
#     end_time = time_to_seconds(clip["end_time"])
#
#     extracted_clip = VideoFileClip(video_path).subclip(start_time, end_time)
#     extracted_clips.append(extracted_clip)
#
# final_clip = concatenate_videoclips(extracted_clips)
#
# output_path = "dataset/Case_D_extracted.MP4"  # Replace with your desired output path
# final_clip.write_videofile(output_path, codec=None)

import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton

class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()

        # 读取本地视频文件
        self.cap = cv2.VideoCapture("dataset/Case_D_extracted.MP4")

        # 检查视频是否成功打开
        if not self.cap.isOpened():
            print("无法打开视频文件")
            sys.exit()

        # 定义界面布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 创建标签用于显示视频帧
        self.label = QLabel()
        self.layout.addWidget(self.label)

        # 创建按钮
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)

        # 连接按钮的点击事件
        self.start_button.clicked.connect(self.start_playing)
        self.stop_button.clicked.connect(self.stop_playing)

        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_playing(self):
        # 开始播放视频
        self.timer.start(33)  # 设置定时器间隔，单位为毫秒（这里约30帧每秒）

    def stop_playing(self):
        # 停止播放视频
        self.timer.stop()

    def update_frame(self):
        # 读取一帧
        ret, frame = self.cap.read()

        # 如果视频已经播放完毕，则停止定时器
        if not ret:
            self.timer.stop()
            return

        # 将视频帧转换为QImage格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

        # 将QImage显示在标签上
        self.label.setPixmap(QPixmap.fromImage(image))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建VideoPlayer实例并显示界面
    video_path = "path/to/your/video.mp4"
    player = VideoPlayer(video_path)
    player.show()

    sys.exit(app.exec_())
