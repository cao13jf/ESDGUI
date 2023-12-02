import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Generate some data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create a figure and plot the data
fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)

# Render the figure to a RGB array
canvas.draw()
width, height = fig.get_size_inches() * fig.get_dpi()
image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

# Display the shape of the generated image array
print(image.shape)