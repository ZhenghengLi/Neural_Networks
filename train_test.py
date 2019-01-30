#!/usr/bin/env python

import sys
import cPickle as pkl
import numpy as np
from network import Network

from network import Network
import mnist

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "USAGE: ", sys.argv[0], "<training_data.pkl>"
        exit(1)

    pkl_file = sys.argv[1]

    print 'loading image data ...'
    with open(pkl_file, 'rb') as fin:
        image_arrays, image_labels = pkl.load(fin)

    training_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[:50000], image_labels[:50000])]
    test_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[50000:], image_labels[50000:])]

    print 'training ...'
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

