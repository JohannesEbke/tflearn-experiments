import tflearn
from models import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np

d1 = AutoEncoder(28*28, [256], 1)
d1.load()
decode = d1.decoder()

# Compare original images with their reconstructions
f, a = plt.subplots(2, 20, figsize=(20, 2))
for i in range(20):
    x = i/20.0
    a[0][i].imshow(np.reshape(decode([x*12-1]), (28, 28)), cmap="hot")
    a[1][i].imshow(np.reshape(decode([(x-0.5)*20]), (28, 28)), cmap="hot")
f.show()
plt.draw()
plt.waitforbuttonpress()

