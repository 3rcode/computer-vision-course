import cv2
import numpy as np
from typing import Tuple

img = cv2.imread("microsoft.png")

def make_kernel(shape: Tuple[int, int]=(5, 5), sigma: float=1.0) -> np.ndarray:
    m, n = shape
    assert m % 2 == 1 and n %2 == 1, "Kernel shape must be odd"
    v = list(range(-(m // 2), (m + 1) // 2, 1))
    h = list(range(-(n // 2), (n + 1) // 2, 1))
    kernel = np.zeros(shape)
    for i in range(m):
        for j in range(n):
            kernel[i, j] = (1 / (2*np.pi*sigma**2)) * np.exp(-(v[i]**2 + h[j]**2) / (2*sigma**2))
    ratio =  np.sum(kernel)
    kernel = kernel / ratio
    return kernel

kernel = make_kernel()

def gaussian_filter(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    m, n = filter.shape
    pad_v = m // 2
    pad_h = n // 2
    convolution = []
    for i in range(3):
        channel = img[:, :, i]
        convol_channel = np.zeros((channel.shape))
        pad_top = np.tile(channel[0, :], (pad_v, 1))
        pad_bot = np.tile(channel[-1, :], (pad_v, 1))
        channel = np.concatenate([pad_top, channel, pad_bot], axis=0)
        pad_left = np.tile(channel[:, 0], (pad_h, 1)).T
        pad_right = np.tile(channel[:, -1], (pad_h, 1)).T
        channel = np.concatenate([pad_left, channel, pad_right], axis=1) 
        x, y = channel.shape
        for v in range(pad_v, x - pad_v):  
            for h in range(pad_h, y - pad_h):  # 
                area = channel[(v - pad_v): (v + pad_v + 1), 
                        (h - pad_h): (h + pad_h + 1)]
                
                convol_channel[v - pad_v, h - pad_h] = np.sum(np.multiply(filter, area))
        
        convolution.append(convol_channel)
    result = np.stack(convolution, axis=2)
    result = result.astype(np.uint8)
    return result

filtered_img = gaussian_filter(img, kernel)

detail = img - filtered_img

sharpen = img + detail * 2
cv2.imshow("Origin", img)
cv2.imshow("Filtered", filtered_img)
cv2.imshow("Detail", detail)
cv2.imshow("Sharpen", sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()
