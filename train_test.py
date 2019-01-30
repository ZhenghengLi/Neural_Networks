#!/usr/bin/env python

import sys
import cPickle as pkl
import numpy as np
from network import Network

from network import Network
import mnist

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "USAGE: ", sys.argv[0], "<training_data.pkl> <test_data.pkl>"
        exit(1)

    traning_data_fn = sys.argv[1]
    test_data_fn = sys.argv[2]

    print 'loading image data ...'
    with open(traning_data_fn, 'rb') as fin:
        image_arrays, image_labels = pkl.load(fin)
    with open(test_data_fn, 'rb') as fin:
        image_arrays_t, image_labels_t = pkl.load(fin)

    training_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[:50000], image_labels[:50000])]
    validation_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[50000:], image_labels[50000:])]
    test_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays_t, image_labels_t)]

    print 'training ...'
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

