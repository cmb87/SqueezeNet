import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

from model.postprocessor import augmentPicture
from model.datapipe import DataPipeline
from model.classifier import Classifier



# Parameters
imageWidth = 120
imageHeight = 120
imageChannels = 3
epocheMax = 100
batchSize = 12
trainpath = '/home/cp/projects/01_machine_learning/90_dataSets/anomalies/concrete/'
testpath = '/home/cp/projects/01_machine_learning/90_dataSets/anomalies/concrete/'
outputDir = './trainingCracks'

# Summary Writer
currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")
trainLogDir = os.path.join(outputDir, currentTime,'train')
testLogDir =  os.path.join(outputDir, currentTime,'test')

trainSummaryWriter = tf.summary.create_file_writer(trainLogDir)
testSummaryWriter = tf.summary.create_file_writer(testLogDir)

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format=("[%(filename)8s] [%(levelname)4s] :  %(funcName)s - %(message)s"),
)

# Build data pipelines
pipeTrain = DataPipeline(
        imagePath=trainpath,
        imageChannels=imageChannels,
        searchPatternImage='*/*.jpg',
        augmentations=["color", "fliph", "crop"]
    )

pipeTrain.build(
    imageSize=(imageHeight,imageWidth),
    batchSize=batchSize, 
    shuffle_buffer_size=600, 
    shuffle=True
)

pipeTest = DataPipeline(
        imagePath=testpath,
        imageChannels=imageChannels,
        searchPatternImage='*/*.jpg',
        augmentations=[],
    )

pipeTest.build(
    imageSize=(imageHeight,imageWidth),
    batchSize=batchSize, 
    shuffle_buffer_size=600, 
    shuffle=False
)

# Build model
assert pipeTest.nClasses == pipeTrain.nClasses, "Number of classes must match!"

classy = Classifier(
    numClasses=pipeTrain.nClasses,
    imageWidth=imageWidth,
    imageHeight=imageHeight,
    imageChannels=imageChannels,
    learnRate=0.001,
)

logging.info(classy.model.summary())

try:
    classy.model.load_weights(os.path.join(outputDir,"weights.h5"))
    logging.info("Weights loaded!")
except:
    logging.warning("Couldnt load weights")

# Model Training 
subitsTrain = np.ceil(pipeTrain.ndata/batchSize)
subitsTest = 1 #np.ceil(pipeTest.ndata/batchSize)
accTrain = tf.keras.metrics.Accuracy()
accTest = tf.keras.metrics.Accuracy()

logImagesIt = 300


for e in range(epocheMax):

    logging.info(f"Starting epoche {e}")

    # =====================================================
    # Training step
    logging.info("Start train")
    for it, (imgs, labels) in enumerate(pipeTrain.ds.take(subitsTrain)):

        # Calculate current global step
        step = e*subitsTrain+it

        # Prepartion for cafe models:
        imgsCaf = 255*imgs - tf.constant([123.68, 116.779, 103.939])

        # Do one training step
        loss, logits = classy.trainStep(imgsCaf, labels)

        # Calculate accuracy
        accTrain.update_state(tf.math.argmax(logits, axis=-1), tf.math.argmax(labels, axis=-1))

        # Write to Tensorboard
        with trainSummaryWriter.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('accuracy', accTrain.result().numpy(), step=step)

        # Logging images
        if step % logImagesIt == 0:

            logging.info("Logging images")

            imgsAug = augmentPicture(imgs, labels, logits, pipeTrain.classes)

            with trainSummaryWriter.as_default():
                tf.summary.image("Augmented train images", imgsAug, step=step, 
                    max_outputs=7, description="Original"
                )

    # =====================================================
    # Run test
    logging.info("Start test")
    lossMean = 0.0
    accMean = 0.0

    for it, (imgs, labels) in enumerate(pipeTest.ds.take(subitsTest)):
        # Prepartion for cafe models:
        imgsCaf = 255*imgs - tf.constant([123.68, 116.779, 103.939])
        loss, logits = classy.testStep(imgsCaf, labels)
        accTest.update_state(tf.math.argmax(logits, axis=-1), tf.math.argmax(labels, axis=-1))
        lossMean += loss.numpy()/subitsTest
        accMean += accTest.result().numpy()/subitsTest


    # Write to Tensorboard
    with testSummaryWriter.as_default():
        tf.summary.scalar('loss', lossMean, step=step)
        tf.summary.scalar('accuracy', accMean, step=step)

    imgsAug = augmentPicture(imgs, labels, logits, pipeTrain.classes)

    with testSummaryWriter.as_default():
        tf.summary.image("Augmented test images", imgsAug, step=step, 
            max_outputs=7, description="Original"
        )


    # =====================================================
    # Storing weights
    classy.model.save_weights(os.path.join(outputDir, "weights.h5"))
    logging.info(f"Weights stored")
