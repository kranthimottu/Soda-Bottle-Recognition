# Soda-Bottle-Recognition

# For data set check the web site below
# http://deepcognition.ai/resources/competitions/soda-bottle-label-detection/

# Modules needed
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, reshape, flatten
from tflearn.layers.estimator import regression
