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

clips = [
    {"start_time": time(hour=0, minute=0, second=50), "end_time": time(hour=0, minute=3, second=54)},
    {"start_time": time(hour=0, minute=10, second=50), "end_time": time(hour=0, minute=14, second=30)},
    {"start_time": time(hour=0, minute=19, second=15), "end_time": time(hour=0, minute=21, second=10)},
    {"start_time": time(hour=0, minute=21, second=10), "end_time": time(hour=0, minute=25, second=0)}
    # Add more clips as needed
]

# Convert the time to seconds
# Convert the time to seconds
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

extracted_clips = []
video_path = "dataset/Case_D.MP4"  # Replace with the path to your video file

for clip in clips:
    start_time = time_to_seconds(clip["start_time"])
    end_time = time_to_seconds(clip["end_time"])

    extracted_clip = VideoFileClip(video_path).subclip(start_time, end_time)
    extracted_clips.append(extracted_clip)

final_clip = concatenate_videoclips(extracted_clips)

output_path = "dataset/Case_D_extracted.MP4"  # Replace with your desired output path
final_clip.write_videofile(output_path, )