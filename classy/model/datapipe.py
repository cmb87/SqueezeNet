import tensorflow as tf
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataPipeline:
    def __init__(self,
        imagePath,
        searchPatternImage,
        imageChannels=3,
        augmentations=["color", "fliph", "crop"],
        *args, **kwargs
    ):

        self.imagePath = imagePath
        self.searchPatternImage = searchPatternImage
        self.imageChannels = imageChannels
        self.augmentations = augmentations
        self.imageSize = None

        # Get class names from folder structure
        self.classes = [d for d in os.listdir(self.imagePath) 
            if os.path.isdir(os.path.join(self.imagePath,d))
        ] # 

        self.classDict = {d:n for n,d in enumerate(self.classes)}

    # ============================
    @property
    def ndata(self):
        return len(glob.glob(self.imagePath+self.searchPatternImage))
    
    @property
    def nClasses(self):
        return len(self.classes)
    

    # ============================
    def build(self, imageSize, batchSize=12, shuffle_buffer_size=600, cache=None, shuffle=False):

        self.imageSize = imageSize

        self.ds = tf.data.Dataset.list_files(self.imagePath+self.searchPatternImage, shuffle=shuffle)
        self.ds = self.ds.map(self._processLoadImage)

        # Augment the image
        if cache:
            if isinstance(cache, str):
                self.ds = self.ds.cache(cache)
            else:
                self.ds = self.ds.cache()

        if shuffle:
            self.ds = self.ds.shuffle(buffer_size=shuffle_buffer_size)
            self.ds = self.ds.repeat()

        # Augment the image
        if 'fliph' in self.augmentations:
            self.ds = self.ds.map(self._processAugmentFlip)
        if 'flipv' in self.augmentations:
            self.ds = self.ds.map(self._processAugmentFlipVertically)
        if 'color' in self.augmentations:
            self.ds = self.ds.map(self._processAugmentColor)
        if 'crop' in  self.augmentations:
            self.ds = self.ds.map(self._processAugmentCrop)

        self.ds = self.ds.map(self._processPrepareForImageNet)

        self.ds = self.ds.batch(batchSize)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)

    # ============================
    def _lookUpClass(self, imgpath):
        # Split the string and take the second last element as class name
        imgClass = tf.strings.split(imgpath, '/')[-2]
        imgClass = imgClass.numpy().decode("utf-8")
        idx = self.classDict[imgClass]
        return idx

    # ============================
    def _processLoadImage(self, imgpath):

        # Read the image
        img = tf.io.read_file(imgpath)
        img = tf.image.decode_jpeg(img, channels=self.imageChannels)

        # Resize the images to a fixed input size, and rescale the input channels to a range of [0,1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, self.imageSize)

        # Since this runs as data pipeline we have to use this to look
        # up the dictionary 
        idx = tf.py_function(self._lookUpClass, [imgpath], tf.int32)

        label = tf.one_hot(idx, len(self.classes))

        return img, label

    # ============================
    def _processPrepareForImageNet(self, img, label):
        #return -1.0+2.0*img, label
        return img, label
    # ============================
    def _processAugmentColor(self, img, label):
        if self.imageChannels == 3:
            img = tf.image.random_hue(img, 0.08)
            img = tf.image.random_saturation(img, 0.6, 1.6)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        return img, label


    # ============================
    def _processAugmentCrop(self, img, label):
        img = tf.image.random_crop( img, size=[self.imageSize[0], self.imageSize[1], self.imageChannels])
        return img, label


    # ============================
    def _processAugmentFlip(self, img, label):
        return tf.image.random_flip_left_right(img), label

    # ============================
    def _processAugmentFlipVertically(self, img, label):
        return tf.image.random_flip_up_down(img), label



if __name__ == '__main__':

    IMAGE_DIR_PATH = '/home/cp/projects/01_machine_learning/90_dataSets/plants/Train/'

    gen = DataPipeline(
            imagePath=IMAGE_DIR_PATH,
            searchPatternImage='*/*.jpg',
        )

    gen.build(imageSize=(120,120))



    for (img, label) in gen.ds.take(5):

        print(img)
        plt.imshow(0.5*(img[0,...]+1))
        plt.show()