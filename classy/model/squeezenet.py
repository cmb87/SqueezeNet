import os
import logging
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    concatenate, Input, Add, Concatenate, GlobalAveragePooling2D, Activation, GaussianNoise



class SqueezeNet:

    imageNetWeights = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        './weights/squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
    )

    imageNetWeightsNoTop = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        './weights/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )

    @staticmethod
    def preprocess(x):
        """When using the pretrained squeezenet, note that this original model
        was trained with cafe style normalization.  The input images should 
        be zero-centered by mean pixel rather than mean image) subtraction. 
        Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].
        Make sure this preprocessing is also used for a new transfer based learning.
        E.g. raw pixel [0,...255] - [103.939, 116.779, 123.68]
        
        Parameters
        ----------
        x : param
            Description
        
        Returns
        -------
        param
            Description
        """
        x -= tf.constant([123.68, 116.779,103.939 ]) 
        return x

    @staticmethod
    def buildModel(imageWidth, imageHeight, imageChannels=3, include_top=True, pretrained=True, lastTrainableLayers=0):

        #SQUEEZE NET V1.1 : A LEX N ET- LEVEL ACCURACY WITH
        #50 X FEWER PARAMETERS AND <0.5MB MODEL SIZE
        inputs = Input((imageHeight, imageWidth, imageChannels))

        n1 = GaussianNoise(5.0)(inputs)

        c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv1", strides=(2, 2))(n1)
        p1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1")(c1)

        f2s1 = Conv2D(16, (1, 1), activation='relu', padding='same', name="Fire2s1")(p1)
        f2e1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire2e1")(f2s1)
        f2e3 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Fire2e3")(f2s1)
        f2 = Concatenate(name="Fire2cat")([f2e1,f2e3])

        f3s1 = Conv2D(16, (1, 1), activation='relu', padding='same', name="Fire3s1")(f2)
        f3e1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire3e1")(f3s1)
        f3e3 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Fire3e3")(f3s1)
        f3 = Concatenate(name="Fire3cat")([f3e1,f3e3])

        p3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool3")(f3)

        f4s1 = Conv2D(32, (1, 1), activation='relu', padding='same', name="Fire4s1")(p3)
        f4e1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="Fire4e1")(f4s1)
        f4e3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Fire4e3")(f4s1)
        f4 = Concatenate(name="Fire4cat")([f4e1,f4e3])

        f5s1 = Conv2D(32, (1, 1), activation='relu', padding='same', name="Fire5s1")(f4)
        f5e1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="Fire5e1")(f5s1)
        f5e3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Fire5e3")(f5s1)
        f5 = Concatenate(name="Fire5cat")([f5e1,f5e3])

        p5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool5")(f5)

        f6s1 = Conv2D(48, (1, 1), activation='relu', padding='same', name="Fire6s1")(p5)
        f6e1 = Conv2D(192, (1, 1), activation='relu', padding='same', name="Fire6e1")(f6s1)
        f6e3 = Conv2D(192, (3, 3), activation='relu', padding='same', name="Fire6e3")(f6s1)
        f6 = Concatenate(name="Fire6cat")([f6e1,f6e3])

        f7s1 = Conv2D(48, (1, 1), activation='relu', padding='same', name="Fire7s1")(f6)
        f7e1 = Conv2D(192, (1, 1), activation='relu', padding='same', name="Fire7e1")(f7s1)
        f7e3 = Conv2D(192, (3, 3), activation='relu', padding='same', name="Fire7e3")(f7s1)
        f7 = Concatenate(name="Fire7cat")([f7e1,f7e3])

        f8s1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire8s1")(f7)
        f8e1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="Fire8e1")(f8s1)
        f8e3 = Conv2D(256, (3, 3), activation='relu', padding='same', name="Fire8e3")(f8s1)
        f8 = Concatenate(name="Fire8cat")([f8e1,f8e3])

        f9s1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire9s1")(f8)
        f9e1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="Fire9e1")(f9s1)
        f9e3 = Conv2D(256, (3, 3), activation='relu', padding='same', name="Fire9e3")(f9s1)
        f9 = Concatenate(name="Fire9cat")([f9e1,f9e3])

        if include_top:
            # It's not obvious where to cut the network... 
            # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
            x = Dropout(0.5, name='drop9')(f9)
            x = Conv2D(1000, (1, 1), padding='valid', name='conv10', activation='relu')(x)
            x = GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Softmax()(x)

        else:
            x = f9

        # Build basic model
        model = Model(inputs=[inputs], outputs=[x], name='squeezenet_extractor')
        logging.info("SqueezeNet built!")

        # Load pretrained weights
        if pretrained and include_top:
            model.load_weights(SqueezeNet.imageNetWeights)
            logging.info("ImageNet weights with top loaded")

        elif pretrained and not include_top:
            model.load_weights(SqueezeNet.imageNetWeightsNoTop)
            logging.info("ImageNet weights without top loaded")

        # Set layers to trainable=False, except for lastTrainableLayers
        for l in range(len(model.layers)-lastTrainableLayers):
            model.layers[l].trainable = False

        return model


if __name__ == '__main__':
    model = SqueezeNet.buildModel(244,244,3, include_top=False, lastTrainableLayers=3)
    #model.trainable = False


    print(model.summary())
    print(model.output)
    print(model.layers[-2].name)


    #for n,l in enumerate(model.layers):
    #    
     #   l.trainable = False
    #    print(n, l.name)