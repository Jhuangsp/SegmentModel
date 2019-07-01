import numpy as np
import cv2

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
 
    if center is None:
        center = (0, 0)
 
    rot = np.identity(3)
    rot[:2, : ] = cv2.getRotationMatrix2D(center, angle, scale)
    M = np.float32([[1, 0, h], [0, 1, 0]])@rot
    rotated = cv2.warpAffine(image, M, (h, w))
 
    return rotated
 
# cap = cv2.VideoCapture('D:\\Astor\\Download\\dataset\\result_20190515\\data_20190515\\squat_front.mp4')
# ret, frame = cap.read()
# (h, w) = frame.shape[:2]
# frame = cv2.resize(frame, (w//2, h//2))
# print(frame.shape)

# cv2.imshow("Original", frame)

# rotated = rotate(frame, -90)
# cv2.imshow("Rotated by 90 Degrees", rotated)
# cv2.waitKey(0)