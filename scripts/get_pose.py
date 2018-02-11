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
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

img=cv2.imread(args.image)
img = misc.imresize(img,(224,224))
img = np.transpose(img, (2,0,1)) # from 224,224,3 to 3,224,224
#img = img[(2,1,0), :, :] # from RGB to BGR
net.blobs['data'].data[0,:,:,:] = img

#print img.shape
out = net.forward()
print out['cls3_fc_xyz']
print out['cls3_fc_wpqr']

#print net.blobs['cls3_fc_wpqr'].data
#print net.blobs['cls3_fc_xyz'].data
