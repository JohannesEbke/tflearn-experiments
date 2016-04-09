import tflearn
from models import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot=True)

# Testing the image reconstruction on new data (test set)
testX = tflearn.data_utils.shuffle(testX)[0]

d = AutoEncoder(28*28, [256], 2)
d.load()

# Applying encode and decode over test set
encode_decode = d.model.predict(testX)

# Compare original images with their reconstructions
f, a = plt.subplots(2, 20, figsize=(20, 2))
for i in range(20):
    a[0][i].imshow(np.reshape(testX[i], (28, 28)), cmap="hot")
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap="hot")
f.show()
plt.draw()
plt.waitforbuttonpress()

