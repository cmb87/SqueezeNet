import numpy as np
import cv2


def augmentPicture(imgs, labels, pred, classNames):

    # Rescale 
    imgs = 255*0.5*(imgs+1)

    # Check if imgs are numpy, else convert it from tf
    if not isinstance(imgs, (np.ndarray, np.generic) ):
        imgs = imgs.numpy()

    # Convert to uint8
    imgs = imgs.astype(np.uint8)

    # Iterate through image batch
    for k in range(imgs.shape[0]):
        classTrue = classNames[np.argmax(labels[k,:])]
        classPred = classNames[np.argmax(pred[k,:])]

        imgs[k,...] = cv2.putText(
            imgs[k,...], "Label:", (5, 10), cv2.FONT_HERSHEY_SIMPLEX,  
            0.3, (0, 255, 0), 1, cv2.LINE_AA
        )
        imgs[k,...] = cv2.putText(
            imgs[k,...], classTrue, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,  
            0.3, (0, 255, 0), 1, cv2.LINE_AA
        )

        imgs[k,...] = cv2.putText(
            imgs[k,...], "Predicted:", (5, 26), cv2.FONT_HERSHEY_SIMPLEX,  
            0.3, (255, 0, 0), 1, cv2.LINE_AA
        )

        imgs[k,...] = cv2.putText(
            imgs[k,...], classPred, (5, 34), cv2.FONT_HERSHEY_SIMPLEX,  
            0.3, (255, 0, 0), 1, cv2.LINE_AA
        )

    return imgs