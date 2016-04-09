from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from os.path import exists

class AutoEncoder(object):
    def __init__(self, input_dimension, layer_dimensions, neck_dimension):
        with tf.Graph().as_default():
            self._input_dimension = input_dimension
            self._layer_dimensions = layer_dimensions
            self._neck_dimension = neck_dimension

            self.input = tflearn.input_data(shape=[None, input_dimension])
            self.layers = [self.input]
            for i, dim in enumerate(layer_dimensions):
                self.layers.append(tflearn.fully_connected(self.layers[-1], dim, name="encoder{}".format(i)))
            self.layers.append(tflearn.fully_connected(self.layers[-1], neck_dimension, name="encoded"))
            self.encoded = self.layers[-1]
            self.decode_index = len(self.layers)
            for i, layer in enumerate(reversed(layer_dimensions)):
                self.layers.append(tflearn.fully_connected(self.layers[-1], layer, name="decoder{}".format(i)))
            self.layers.append(tflearn.fully_connected(self.layers[-1], input_dimension, name="decoded"))
            self.decoded = self.layers[-1]
            self.regression = tflearn.regression(self.decoded, optimizer='adam', learning_rate=0.001,
                                                 loss='mean_square', metric=None)
            self.model = tflearn.DNN(self.regression, tensorboard_verbose=0)

    def learn(self, X, testX):
        with self.model.net.graph.as_default():
            self.model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
                           run_id=repr(self), batch_size=256)

    def __repr__(self):
        dims = [self._input_dimension] + self._layer_dimensions + [self._neck_dimension]
        return "auto_encoder_{}".format("_".join(map(str, dims)))

    def save(self, filename=None):
        if filename is None:
            filename = repr(self)
        with self.model.net.graph.as_default():
            self.model.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = repr(self)
        with self.model.net.graph.as_default():
            self.model.load(filename)

    def decoder(self):
        with tf.Graph().as_default():
            print(self._neck_dimension)
            input_layer = tflearn.input_data(shape=[None, self._neck_dimension])
            layers = [input_layer]
            for i, dim in enumerate(reversed(self._layer_dimensions)):
                print(dim)
                layers.append(tflearn.fully_connected(layers[-1], dim, name="decoder{}".format(i)))
            print(self._input_dimension)
            decoder = tflearn.fully_connected(layers[-1], self._input_dimension, name="decoded")
            layers.append(decoder)
            regression = tflearn.regression(decoder)
            decode_model = tflearn.DNN(decoder)
            for i, layer in enumerate(layers[1:]):
                decode_model.set_weights(layer.W, self.model.get_weights(self.layers[self.decode_index+i].W))
                decode_model.set_weights(layer.b, self.model.get_weights(self.layers[self.decode_index+i].b))
            def decode(v):
                return decode_model.predict([v])
            return decode
