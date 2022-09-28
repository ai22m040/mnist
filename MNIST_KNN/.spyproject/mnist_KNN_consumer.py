# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:12:26 2022

@author: kloimstg
"""

import pickle
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

# Open the image form working directory
image = Image.open('Zahl10.jpg')
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)

image.show()

# convert to greyscale
image = image.convert("L")

#risize the image
image = image.resize((28,28),resample=Image.Resampling.NEAREST )
image.show()

# convert image to numpy array
imageData = np.array(image)
# summarize shape
print(imageData.shape)


# flatten the image (reshape array)
flattenedImage = imageData.reshape(1,784)

# Loading the model
filename = 'mnist_knn_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# make prediction
number = loaded_model.predict(flattenedImage)

print("\nI think the image shows number ", number)

