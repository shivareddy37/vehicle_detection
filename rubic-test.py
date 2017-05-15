
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import tkinter
import math
# from utils import *
# vidcap = cv2.VideoCapture('project_video.mp4')
# count = 1
# while True:
#     success, image = vidcap.read()
#     if not success:
#         print('code not open')
#         break
#     cv2.imwrite('video_images/img' + str(count) +'.png' , image)     # save frame as JPEG file
#     count += 1
#     print('done')
a = [((23,45), (45,78)), ((34,451), (452,728)), ((123,425), (445,784))]
b = [((123,445), (465,788)), ((234,4511), (1452,7128)), ((1323,4245), (5445,7584))]
c = a+b
print(c)