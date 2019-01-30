#!/usr/bin/env python

import sys
import cPickle as pkl
import numpy as np

from network import Network
import mnist

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "USAGE: ", sys.argv[0], "<image_array> <image_label>"
        exit(1)

    image_arrays_fn = sys.argv[1]
    image_labels_fn = sys.argv[2]

    print 'loading image data ...'
    image_arrays = mnist.load_MNIST_images(image_arrays_fn)
    image_labels = mnist.load_MNIST_labels(image_labels_fn)

    with open('training_data.pkl', 'wb') as fout:
        pkl.dump( (image_arrays, image_labels), fout )

    training_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[:50000], image_labels[:50000])]
    test_data = [mnist.one_training_tuple(image_array, digit)
            for image_array, digit in zip(image_arrays[50000:], image_labels[50000:])]

    print 'ready to train.'

