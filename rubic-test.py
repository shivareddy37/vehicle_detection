# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
#
# image = cv2.imread('test_images/test5.jpg')
#
# rhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
# ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
# bhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
#
# bin_edges = rhist[1]
# bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
#
# print(bin_centers)
#
# fig = plt.figure(figsize=(12,3))
# plt.subplot(131)
# plt.bar(bin_centers, rhist[0])
# plt.xlim(0, 256)
# plt.title('R Histogram')
# plt.subplot(132)
# plt.bar(bin_centers, ghist[0])
# plt.xlim(0, 256)
# plt.title('G Histogram')
# plt.subplot(133)
# plt.bar(bin_centers, bhist[0])
# plt.xlim(0, 256)
# plt.title('B Histogram')
# plt.show()
import pandas
from sklearn.preprocessing import StandardScaler

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from utils import *
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit()


# def plot3d(pixels, colors_rgb,
#         axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
#     """Plot pixels in 3D."""
#
#     # Create figure and 3D axes
#     fig = plt.figure(figsize=(8, 8))
#     ax = Axes3D(fig)
#
#     # Set axis limits
#     ax.set_xlim(*axis_limits[0])
#     ax.set_ylim(*axis_limits[1])
#     ax.set_zlim(*axis_limits[2])
#
#     # Set axis labels and sizes
#     ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
#     ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
#     ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
#     ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)
#
#     # Plot pixel values with colors given in colors_rgb
#     ax.scatter(
#         pixels[:, :, 0].ravel(),
#         pixels[:, :, 1].ravel(),
#         pixels[:, :, 2].ravel(),
#         c=colors_rgb.reshape((-1, 3)), edgecolors='none')
#
#     return ax  # return Axes3D object for further manipulation
#
#
# # Read a color image
# img = cv2.imread('test_images/test5.jpg')
#
# # Select a small fraction of pixels to plot by subsampling it
# scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
# img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
#
# # Convert subsampled image to desired color space(s)
# img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
# img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
# img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
#
# # Plot and show
# plot3d(img_small_RGB, img_small_rgb)
# plt.show()
#
# plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
# plt.show()