import tflearn
from models import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
import tflearn.datasets.mnist as mnist
import sys

layers = list(map(int, sys.argv[1:]))
neck = layers[-1]
layers = layers[:-1]

X, Y, testX, testY = mnist.load_data(one_hot=True)

# Testing the image reconstruction on new data (test set)
#Xes = tflearn.data_utils.shuffle(testX)[0]
Xes = tflearn.data_utils.shuffle(testX)[0]

d = AutoEncoder(28*28, layers, neck)
d.load()

# Applying encode and decode over test set
encode_decode = d.model.predict(Xes)

# Compare original images with their reconstructions
f, a = plt.subplots(3, 20, figsize=(20, 3))
for i in range(20):
    a[0][i].imshow(np.reshape(Xes[i], (28, 28)), cmap="hot")
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap="hot")
    a[2][i].imshow(np.reshape(Xes[i]-encode_decode[i], (28, 28)), cmap="hot")
f.show()
plt.draw()
plt.waitforbuttonpress()

