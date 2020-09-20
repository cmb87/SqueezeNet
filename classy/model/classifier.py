import tensorflow as tf
import logging
import numpy as np
from .squeezenet import SqueezeNet
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    concatenate, Input, Add, Concatenate, Dense, Flatten, GlobalAveragePooling2D



class Classifier:

    def __init__(
            self, numClasses, imageWidth, imageHeight, imageChannels,
            lastTrainableLayers=0, learnRate=0.001, featureExtractor="SqueezeNet"
    ):

        # Build the model
        self.model = self._buildModel(
            numClasses, imageWidth,
            imageHeight, imageChannels,
            featureExtractor,
            lastTrainableLayers,
        )

        # Pick an optimizer
        self.optimizer = tf.keras.optimizers.Adam(learnRate)



    def _buildModel(self, numClasses, imageWidth, imageHeight, imageChannels, featureExtractor, lastTrainableLayers=0):

        if featureExtractor == "MobileNetV2":
            baseModel = tf.keras.applications.MobileNetV2(
                input_shape=[imageHeight, imageWidth, imageChannels],
                include_top=False,
                weights='imagenet'
            )

            # Set layers to trainable=False, except for lastTrainableLayers
            for l in range(len(baseModel.layers) - lastTrainableLayers):
                baseModel.layers[l].trainable = False
            logging.info("Use MobileNetV2")

        elif featureExtractor == "SqueezeNet":
            baseModel = SqueezeNet.buildModel(
                imageWidth, imageHeight, imageChannels=3,
                include_top=False, pretrained=True, lastTrainableLayers=lastTrainableLayers
            )
            logging.info("Use SqueezeDet")

        # Classification head
        x = Dropout(0.5, name='drop9')(baseModel.output)
        x = Conv2D(numClasses, (1, 1), padding='valid', name='conv10', activation='relu')(x)
        logits = GlobalAveragePooling2D()(x)

        # Classical Classification head
        #x = Flatten()(baseModel.output)
        #x = Dropout(0.5)(x)
        #x = Dense(256, activation="relu")(x)
        #x = Dropout(0.5)(x)
        #logits = Dense(numClasses)(x)

        model = Model(inputs=[baseModel.input], outputs=logits)

        return model


    def loss(self, labels, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


    @tf.function
    def trainStep(self, imgs, labels):
        # Calculate gradients of loss wrt to imgs
        with tf.GradientTape() as t:
            logits = self.model(imgs)
            loss = self.loss(labels, logits)

        # Change weights
        grads = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, logits

    @tf.function
    def testStep(self, imgs, labels):
        logits = self.model(imgs)
        loss = self.loss(labels, logits)
        return loss, logits
