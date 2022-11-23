import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('tiger.jpg', 0)
img2 = cv2.imread('lion.jpg', 0)
img3 = cv2.imread('elephant.jpg', 0)
img4 = cv2.imread('deer.jpg', 0)

images = [img1, img2, img3, img4]
labels = ['tiger', 'lion', 'elephant', 'deer']

plt.figure(figsize=(10, 10))
for x in range(len(images)):
    plt.subplot(2, 2, x+1)
    plt.imshow(images[x], cmap='gray')
plt.show()

med = []
for i in images:
    m = cv2.medianBlur(i, 5)
    med.append(m)


#OTSU thresholding
bin = []
for m in med:
    ret, otsu = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bin.append(otsu)

plt.figure(figsize=(10, 10))
for x in range(len(images)):
    plt.subplot(2, 2, x+1)
    plt.imshow(bin[x], cmap='gray')
plt.show()

import numpy as np

kernel = np.ones((11, 11), np.uint8)

closed = []
for b in bin:
    c = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    closed.append(c)

plt.figure(figsize=(10, 10))
for x in range(len(images)):
    plt.subplot(2, 2, x+1)
    plt.imshow(closed[x], cmap='gray')
plt.show()

plt.figure(figsize=(10, 10))
for x in range(len(images)):
    plt.subplot(2, 2, x+1)
    d = cv2.dilate(closed[x], kernel, iterations=2)
    e = cv2.erode(d, kernel, iterations=2)
    plt.imshow(e, cmap='gray')
plt.show()

cv2.waitKey()