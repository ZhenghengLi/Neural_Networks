#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def _lbtoi(bs):
    return int(bs[::-1].encode('hex'), 16)

def _bbtoi(bs):
    return int(bs.encode('hex'), 16)

def load_MNIST_images(filename):
    image_arrays = []
    with open(filename, 'rb') as fin:
        num_magic = _bbtoi(fin.read(4))
        num_images = _bbtoi(fin.read(4))
        num_rows = _bbtoi(fin.read(4))
        num_cols = _bbtoi(fin.read(4))
        for idx in xrange(num_images):
            pixel_bytes = fin.read(num_rows * num_cols)
            pixels = [_bbtoi(b) / 255.0 for b in pixel_bytes]
            image_arrays.append(pixels)
    return np.array(image_arrays, dtype = np.float32)

def load_MNIST_labels(filename):
    image_labels = None
    with open(filename, 'rb') as fin:
        num_magic = _bbtoi(fin.read(4))
        num_items = _bbtoi(fin.read(4))
        item_bytes = fin.read(num_items)
        image_labels = [_bbtoi(b) for b in item_bytes]
    return np.array(image_labels, dtype = np.int8)

def digit_vector(digit):
    vec = np.zeros((10, 1))
    vec[digit] = 1.0
    return vec

def show_one_image(image_arrays, image_labels, idx):
    image_array = image_arrays[idx]
    image_label = image_labels[idx]
    print 'the digit = ', image_label
    plt.imshow(image_array.reshape(28, 28))
    plt.show()

