import cv2
import scipy.signal as sig
import numpy as np

image = np.asarray([[ 7,  6,  7,  8,  9, 10,  9],
                    [ 2,  1,  2,  3,  4,  5,  4],
                    [ 7,  6,  7,  8,  9, 10,  9],
                    [12, 11, 12, 13, 14, 15, 14],
                    [17, 16, 17, 18, 19, 20, 19],
                    [12, 11, 12, 13, 14, 15, 14]])


kernel = np.ones((3, 3)) / 9

after = sig.convolve2d(image, kernel, mode="valid")

print(after)
