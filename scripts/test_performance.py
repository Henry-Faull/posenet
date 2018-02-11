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

    #standard image: 1920, 1080

    img = cv2.imread(image)
    #img = misc.imresize(img,(224,224))

    #resize image to 256px in the smaller dimension
    height = img.shape[0]
    width = img.shape[1]

    if height > width:
        r = 256.0 / width
        dim = (256,int(height * r))
    else:
        r = 256.0 / height
        dim = (int(width * r), 256)

    reImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    reHeight = reImg.shape[0]
    reWidth = reImg.shape[1]

    #crop to 224pxx224px
    #assume width > height
    reWidth = reWidth - reWidth%2
    diff = (reWidth - 224)/2
    cropImg = reImg[16:240,diff:(reWidth-diff)]

    cv2.imwrite("imagedump/cropped" + str(i) + ".png",cropImg,[])

    finImg = np.transpose(cropImg, (2,0,1))
    #cropImg = np.transpose(cropImg, (2,0,1)) # from 224,224,3 to 3,224,224
    #img = img[(2,1,0), :, :] # from RGB to BGR

    net.blobs['data'].data[0,:,:,:] = finImg

    print pose_x
    print pose_q

    out = net.forward()
    #predicted_x = out['cls3_fc_xyz']
    #predicted_q = out['cls3_fc_wpqr']

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

    print 'Iteration:  ', i-3, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta
    print '------------------------------------------------------------------------------------------'


median_result = np.median(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

np.savetxt('results-performance.txt', results, delimiter=' ')

print 'Success!'
