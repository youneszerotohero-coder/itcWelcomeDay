FLIP_CAMERA=False
import cv2

from config import FLIP_CAMERA 
if FLIP_CAMERA:
    frame = cv2.flip(frame,1)