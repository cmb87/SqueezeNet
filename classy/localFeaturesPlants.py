import os
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras import Model
from datetime import datetime
import matplotlib.pyplot as plt

from model.postprocessor import augmentPicture
from model.datapipe import DataPipeline
from model.classifier import Classifier

# Set logging formats
logging.basicConfig(
    level=logging.INFO,#{"info": logging.INFO, "critical": logging.CRITICAL}[flags.loggingLevel],
    format=("[%(filename)8s] [%(levelname)4s] :  %(funcName)s - %(message)s"),
)

# Parameters
imageWidth = 120
imageHeight = 120
imageChannels = 3
epocheMax = 100
batchSize = 1
trainpath = '/home/cp/projects/01_machine_learning/90_dataSets/plants/Train/'
testpath = '/home/cp/projects/01_machine_learning/90_dataSets/plants/Test/'
outputDir = './trainingPlants'

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format=("[%(filename)8s] [%(levelname)4s] :  %(funcName)s - %(message)s"),
)

# Build data pipelines
pipeTest = DataPipeline(
        imagePath=testpath,
        imageChannels=imageChannels,
        searchPatternImage='*/*.jpg',
        augmentations=[],
)
pipeTest.build(
    imageSize=(imageHeight,imageWidth),
    batchSize=batchSize,
    shuffle_buffer_size=200,
    shuffle=True
)

# Build model
classy = Classifier(
    numClasses=pipeTest.nClasses,
    imageWidth=imageWidth,
    imageHeight=imageHeight,
    imageChannels=imageChannels,
    learnRate=0.001,
    lastTrainableLayers=2,
)

logging.info(classy.model.summary())

classy.model.load_weights(os.path.join(outputDir,"weights.h5"))
logging.info("Weights loaded!")



intermediate = Model(inputs=[classy.model.input], outputs=classy.model.layers[-2].output)

print(intermediate.summary())
for it, (imgs, labels) in enumerate(pipeTest.ds.take(31)):
    imgsCaf = 255 * imgs - tf.constant([123.68, 116.779, 103.939])



    #probs = tf.nn.softmax(classy.model.predict(imgsCaf),-1)
    #idx = tf.argmax(probs, axis=-1)
    #idxTrue = tf.argmax(labels, axis=-1)
    #convout = intermediate.predict(imgsCaf)[..., idx]

    probs = tf.nn.softmax(classy.model.predict(imgsCaf), -1)
    idx = tf.argmax(probs, axis=-1)
    idxTrue = tf.argmax(labels, axis=-1)
    #convout = intermediate.predict(imgsCaf)
    #convout = tf.matmul(convout, tf.transpose(probs))
    print(idx)
    convout = tf.expand_dims(intermediate.predict(imgsCaf)[...,idx[0]],-1)

    print(probs)
    print(convout.shape)
    anomalies = tf.image.resize(convout, (imageHeight, imageWidth))

    anomalies = (anomalies - tf.reduce_min(anomalies))/(tf.reduce_max(anomalies)-tf.reduce_min(anomalies))

    plt.title(f"Pred {idx} True {idxTrue}")
    plt.imshow(imgs[0,...])
    plt.imshow(anomalies[0,:,:,0], cmap="jet", alpha=0.3)
    plt.show()