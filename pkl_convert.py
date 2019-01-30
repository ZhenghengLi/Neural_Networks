#!/usr/bin/env python

import sys
import cPickle as pkl

from network import Network
import mnist

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "USAGE: ", sys.argv[0], "<image_array> <image_label> <pkl_file>"
        exit(1)

    image_arrays_fn = sys.argv[1]
    image_labels_fn = sys.argv[2]
    pkl_fn = sys.argv[3]

    print 'loading image data ...'
    image_arrays = mnist.load_MNIST_images(image_arrays_fn)
    image_labels = mnist.load_MNIST_labels(image_labels_fn)

    print 'writing to pkl file ...'
    with open(pkl_fn, 'wb') as fout:
        pkl.dump( (image_arrays, image_labels), fout, protocol = pkl.HIGHEST_PROTOCOL)

