import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import cv2
from scipy import misc
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

# Make sure that caffe is on the python path:
caffe_root = ''  # Change to your directory to caffe-posenet
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

dataFile = open(args.dataset,'r')
lines = dataFile.readlines()

filePrefix = '../Cambridge-Landmarks/KingsCollege/'

results = np.zeros((len(lines),2))

for i in range(0,len(lines)):
    lineData = lines[i].split(' ')
    if len(lineData) != 8:
        continue
    image = filePrefix + lineData[0]
    pose_x = lineData[1:4]
    pose_q = lineData[4:8]
    pose_x = [float(k) for k in pose_x]
    pose_q = [float(m) for m in pose_q]

    image = caffe.io.load_image(image)

    #mu = np.load(caffe_root + '../Cambridge-Datasets/kingscollege_train_mean.npy')
    #mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    #transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1, 3, 224, 224)

    net.blobs['data'].data[0,:,:,:] = transformer.preprocess('data', image)

    cv2.imwrite("imagedump/im" + str(i) + ".png",image,[])

    # perform classification
    net.forward()

    print pose_x
    print pose_q

    #out = net.forward()

    predicted_q = net.blobs['cls3_fc_wpqr'].data
    predicted_x = net.blobs['cls3_fc_xyz'].data

    predicted_x = np.squeeze(predicted_x)
    predicted_q = np.squeeze(predicted_q)

    print predicted_x
    print predicted_q

    #Compute Individual Sample Error
    q1 = pose_q / np.linalg.norm(pose_q)
    q2 = predicted_q / np.linalg.norm(predicted_q)
    d = abs(np.sum(np.multiply(q1,q2)))
    theta = 2 * np.arccos(d) * 180/math.pi
    error_x = np.linalg.norm(pose_x-predicted_x)

    results[i,:] = [error_x,theta]

    print 'Image:  ', i-3, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta
    print '------------------------------------------------------------------------------------------'


median_result = np.median(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

np.savetxt('results-performance.txt', results, delimiter=' ')

print 'Success!'
