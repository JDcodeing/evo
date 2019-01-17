import numpy as np
import cv2
import matplotlib.pyplot as plt

# Source gradient image from 0 to 255
src = np.atleast_2d(np.linspace(0,255,10));

# Set up to interpolate from first pixel value to last pixel value
map_x_32 = np.linspace(0,9,101)
map_x_32 = np.atleast_2d(map_x_32).astype('float32')
map_y_32 = map_x_32*0

# Interpolate using OpenCV
output = cv2.remap(src, map_x_32, map_y_32, cv2.INTER_LINEAR)

# Truth
output_truth = np.atleast_2d(np.linspace(0,255,101));

interp_error = output - output_truth

plt.plot(interp_error[0])